import random
import numpy as np
from collections import defaultdict
import itertools
import random
import math

class RiskAverseQLearning:
    def __init__(self, K, N, M, I, Ts, d):
        self.num_devices = K
        self.num_sub6 = N
        self.num_mmWave = M
        self.num_Qtable = I
        self.frame_duration = Ts
        self.packet_size = d
        
        self.exploration_rate = 0.5 # eps
        self.decay_factor = 0.995 # lambda
        self.discount_factor = 0.9 # gamma
        
        self.risk_control = 0.5 # lambda_p
        self.utility_func_param = -0.5 # beta
        
        self.PLR_req = 0.1 # phi_max
        self.Lk = [6] * K
        
        self.known_average_rate = [[0,0] for _ in range(K)]
        self.OldSuccess = [[0, 0] for _ in range(K)]
        self.Success = [[0, 0] for _ in range(K)]
        self.Alloc = [[0, 0] for _ in range(K)]
        self.PLR = [[0.0, 0.0] for _ in range(K)]
        self.PSR = [1.0 for _ in range(K)]
        self.Q_table = [defaultdict(lambda:defaultdict(lambda:0.0)) for _ in range(I)]
        self.Q_sum = defaultdict(lambda:defaultdict(lambda:0.0))
        self.Q_hat = [defaultdict(lambda:defaultdict(lambda:0.0)) for _ in range(I)]
        self.Count = [defaultdict(lambda:defaultdict(int)) for _ in range(I)]
        self.cur_state = None
        self.init_state()
    
    def init_state(self):
        state = []
        for k in range(self.num_devices):
            a = random.choice(range(self.Lk[k]+1))
            b = random.choice(range(self.Lk[k]+1))
            state.extend([
                random.choice([0,1]),
                random.choice([0,1]),
                a,
                b
            ])
        self.cur_state = tuple(state)
    
    def update_state(self):
        state = []
        for k in range(self.num_devices):
            state.extend([
                int(self.PLR[k][0] <= self.PLR_req),
                int(self.PLR[k][1] <= self.PLR_req),
                self.OldSuccess[k][0],
                self.OldSuccess[k][1]
                ])
        self.cur_state = tuple(state)
    
    def get_random_action_tuple(self):
        return tuple(random.choices(range(3), k=self.num_devices))
    
    def get_action_tuple(self, Q_hat_index):
        Q_hat_ = self.Q_hat[Q_hat_index]
        cur_state = self.cur_state
        return self.get_max_action(Q_hat_, cur_state)[0]
        
        # compute Q_hat explicitly
        Q_bar = defaultdict(lambda:defaultdict(lambda:0.0))
        for i in range(self.num_Qtable):
            if cur_state not in self.Q_table[i]:
                continue
            q = self.Q_table[i][cur_state]
            for a in q:
                Q_bar[cur_state][a] += q[a]
        Q_hat = defaultdict(lambda:defaultdict(lambda:0.0))
        for a in itertools.product(range(3), repeat=self.num_devices):
            for i in range(self.num_Qtable):
                if cur_state in self.Q_table[i] and a in self.Q_table[i][cur_state]:
                    Q_hat[cur_state][a] += (self.Q_table[i][cur_state][a] - Q_bar[cur_state][a] / self.num_Qtable)**2
            q = 0
            if cur_state in self.Q_table[Q_hat_index] and a in self.Q_table[Q_hat_index][cur_state]:
                q = self.Q_table[Q_hat_index][cur_state][a]
            Q_hat[cur_state][a] = q - self.risk_control / max(1,(self.num_Qtable-1)) * Q_hat[cur_state][a]
            
        # sanity check
        if cur_state in Q_hat_:
            for a in Q_hat_[cur_state]:
                assert abs(Q_hat_[cur_state][a] - Q_hat[cur_state][a]) <= 1e-6, f"Expecting {Q_hat_[cur_state][a]}, got {Q_hat[cur_state][a]}"
        return self.get_max_action(Q_hat, cur_state)[0]
    
    def receive_reward(self, reward, cur_frame, sample_achievable):
        # reward = array of [success sub6, success mmWave]
        self.OldSuccess = self.Success
        self.Success = reward
        
        total_reward = 0

        # update PLR
        for k in range(self.num_devices):
            #update based on current, not new state
            total_reward -= 1 - int(self.PLR[k][0] <= self.PLR_req)
            total_reward -= 1 - int(self.PLR[k][1] <= self.PLR_req)
            
            for i in range(2):
                last_plr = self.PLR[k][i] * (cur_frame-1)
                new_plr = 0
                if self.Alloc[k][i] != 0:
                    new_plr = 1 - self.Success[k][i] / self.Alloc[k][i]
                self.PLR[k][i] = (last_plr + new_plr) / cur_frame
        #print("PLR each device: ", [sum(x)/2 for x in self.PLR])
        print("PLR each device: ", self.PLR)
        
        # update PSR
        for k in range(self.num_devices):
            last_psr = self.PSR[k] * (cur_frame-1)
            new_psr = 1
            if sum(self.Alloc[k]) != 0:
                new_psr = sum(self.Success[k]) / sum(self.Alloc[k])
            self.PSR[k] = (last_psr + new_psr) /  cur_frame
            
            total_reward += self.PSR[k]
        print("PSR each device: ", self.PSR)
        
        # update known average rate
        for k in range(self.num_devices):
            for i in range(2):
                if not self.Alloc[k][i]:
                    continue
                old_rate = self.known_average_rate[k][i] * (cur_frame-1)
                new_rate = sample_achievable[k][i]
                self.known_average_rate[k][i] = (old_rate + new_rate) / cur_frame
                
        return total_reward
    
    def map_action(self, action_chosen):
        for i, action in enumerate(action_chosen):
            if action == 0:
                self.Alloc[i][0] = max(1, min(int(self.known_average_rate[i][0] * self.frame_duration / self.packet_size), self.Lk[i]))
                self.Alloc[i][1] = 0
            elif action == 1:
                self.Alloc[i][0] = 0
                self.Alloc[i][1] = max(1, min(int(self.known_average_rate[i][1] * self.frame_duration / self.packet_size), self.Lk[i]))
            else:
                self.Alloc[i][1] = max(1, min(int(self.known_average_rate[i][1] * self.frame_duration / self.packet_size), self.Lk[i]))
                self.Alloc[i][0] = max(1, min(int(self.known_average_rate[i][0] * self.frame_duration / self.packet_size), self.Lk[i] - self.Alloc[i][1]))
        return self.Alloc
    
    def get_max_action(self, Q, s):
        if s not in Q:
            return [self.get_random_action_tuple(), 0]
        state = Q[s]
        curMaxNegAction = [None, -1e9]
        curMaxPosAction = [None, -1e9]
        negActionCount = 0
        for action in itertools.product(range(3), repeat=self.num_devices):
            if action not in Q:
                continue
            v = state[action]
            if v < 0:
                if v > curMaxNegAction[1]:
                    curMaxNegAction = [action, v]
                negActionCount += 1
            else:
                if v > curMaxPosAction[1]:
                    curMaxPosAction = [action, v]
        if curMaxNegAction[0] is None and curMaxPosAction[0] is None:
            return [self.get_random_action_tuple(), 0]
        if curMaxPosAction[0] is not None:
            return curMaxPosAction
        if negActionCount == 3**self.num_devices:
            return curMaxNegAction
        aList = []
        for action in itertools.product(range(3), repeat=self.num_devices):
            if action not in Q:
                aList.append(action)
        return [random.choice(aList), 0]
    
    def get_current_action(self):
        Q_hat_chosen = random.choice(range(self.num_Qtable))
        print("Q_H: ", Q_hat_chosen)
        self.exploration_rate *= self.decay_factor
        r1 = random.random()
        if r1 < self.exploration_rate:
            action_chosen = self.get_random_action_tuple()
        else:
            action_chosen = self.get_action_tuple(Q_hat_chosen)
            
        A = sum(k in (0, 2) for k in action_chosen)
        B = sum(k in (1, 2) for k in action_chosen)
            
        return action_chosen, (A <= self.num_sub6 and B <= self.num_mmWave)
        
    def calc_utility(self, x):
        return -math.exp(self.utility_func_param * x) 
        
    def update_to_new_state(self, reward, cur_frame, action, sample_achievable_rate):
        assert(self.cur_state)
        old_state = tuple(list(self.cur_state))
        
        rew = self.receive_reward(reward, cur_frame, sample_achievable_rate)
        print("Reward: ", rew)
        self.update_state()
        
        new_state = self.cur_state
        print("Old state: ", old_state)
        print("New state: ", new_state)
        
        # update table
        msk = np.random.poisson(size=self.num_Qtable)
        for i, v in enumerate(msk):
            if v != 1:
                continue
            oldQ = 0
            if old_state in self.Q_table[i] and action in self.Q_table[i][old_state]:
                oldQ = self.Q_table[i][old_state][action]
            oldA = 0
            if old_state in self.Count[i] and action in self.Count[i][old_state]:
                oldA = 1 / self.Count[i][old_state][action]
            
            # find max Action in current Q (max[a] Q(s(t+1), *))
            max_QA = self.get_max_action(self.Q_table[i], new_state)[1]
            x0 = -1
            newQ = oldQ + oldA * (self.calc_utility(rew + self.discount_factor * max_QA - oldQ) - x0)
            
            self.Count[i][old_state][action] += 1 # line 14+15
            
            if oldQ != newQ:
                if newQ != 0:
                    self.Q_table[i][old_state][action] = newQ
                C = self.Q_sum[old_state][action] - oldQ
                self.Q_sum[old_state][action] += newQ-oldQ
                factor_rate = -self.risk_control / max(1,(self.num_Qtable-1))
                inside_change = newQ*newQ - oldQ*oldQ
                inside_change -= ((newQ-oldQ)*(newQ+oldQ+C+C)) / self.num_Qtable
                for j in range(self.num_Qtable): # update all Q_hat
                    self.Q_hat[j][old_state][action] += factor_rate * inside_change
                    if i == j:
                        self.Q_hat[j][old_state][action] += newQ - oldQ
        
        # sanity check
        #for i in range(self.num_Qtable):
        #   sanity_check = sum(self.Q_table[a][old_state][action] for a in range(self.num_Qtable))
        #   assert abs(self.Q_sum[old_state][action] - sanity_check) <= 1e-6, f"Expecting {sanity_check}, got {self.Q_sum[old_state][action]}"
        # 
        #   sanity_check = self.Q_table[i][old_state][action] if old_state in self.Q_table[i] and action in self.Q_table[i][old_state] else 0
        #   qavg = sum([self.Q_table[a][old_state][action] if old_state in self.Q_table[a] and action in self.Q_table[a][old_state] else 0 for a in range(self.num_Qtable)]) / self.num_Qtable
        #   
        #   sanity_check -= self.risk_control / max(1,(self.num_Qtable-1)) * sum([(self.Q_table[a][old_state][action]-qavg)**2 if old_state in self.Q_table[a] and action in self.Q_table[a][old_state] else qavg**2 for a in range(self.num_Qtable)])
        #   assert abs(self.Q_hat[i][old_state][action] - sanity_check) <= 1e-6, f"Expecting {sanity_check}, got {self.Q_hat[i][old_state][action]}"
        
        return rew