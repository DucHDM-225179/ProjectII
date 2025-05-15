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
        self.Success = [[0, 0] for _ in range(K)]
        self.Alloc = [[0, 0] for _ in range(K)]
        self.PLR = [[0.0, 0.0] for _ in range(K)]
        self.PSR = [1.0 for _ in range(K)]
        self.Q_table = [defaultdict(lambda:defaultdict(lambda:0.0)) for _ in range(I)]
        self.Count = [defaultdict(lambda:defaultdict(int)) for _ in range(I)]
        self.cur_state = None
        self.init_state()
    
    
        self.CC = defaultdict(int)
    
    def init_state(self):
        """
        Initialize to a random state
        """
        state = []
        for k in range(self.num_devices):
            a = random.choice(range(self.Lk[k]+1))
            b = random.choice(range(self.Lk[k]+1))
            state.extend([
                random.choice([0,1]),
                random.choice([0,1]),
                a,
                self.Lk[k]-a
            ])
        self.cur_state = tuple(state)
    
    def update_state(self):
        """
        Update the current state based on PLR and Success
        """
        state = []
        for k in range(self.num_devices):
            state.extend([
                int(self.PLR[k][0] <= self.PLR_req),
                int(self.PLR[k][1] <= self.PLR_req),
                self.Success[k][0],
                self.Success[k][1]
                ])
        self.cur_state = tuple(state)
    
    def get_random_action_tuple(self):
        """
        Get a random {x}^k tuple, x in (0,1,2)
        """
        return tuple(random.choices(range(3), k=self.num_devices))
    
    def get_action_tuple(self, Q_hat_index):
        """
        Get an action when choose Q^H = Q_hat_index
        """
        cur_state = self.cur_state
        
        # compute Q_hat explicitly
        Q_bar = defaultdict(lambda:defaultdict(lambda:0.0))
        for i in range(self.num_Qtable):
            if cur_state not in self.Q_table[i]:
                continue
            q = self.Q_table[i][cur_state]
            for a in q:
                Q_bar[cur_state][a] += q[a]
                
        Q_hat = defaultdict(lambda:defaultdict(lambda:0.0))
        mx = 0
        for a in itertools.product(range(3), repeat=self.num_devices):
            # variance
            Q = Q_bar[cur_state][a] / self.num_Qtable
            for i in range(self.num_Qtable):
                qval = 0
                if cur_state in self.Q_table[i] and a in self.Q_table[i][cur_state]:
                    qval = self.Q_table[i][cur_state][a]
                Q_hat[cur_state][a] += (qval - Q)**2
                            
            # Q_hat
            q = 0
            if cur_state in self.Q_table[Q_hat_index] and a in self.Q_table[Q_hat_index][cur_state]:
                q = self.Q_table[Q_hat_index][cur_state][a]
            V = q - self.risk_control / max(1,(self.num_Qtable-1)) * Q_hat[cur_state][a]
            Q_hat[cur_state][a] = V
            mx = max(mx, V)
        a, v = self.get_max_action(Q_hat, cur_state)
        #assert(mx >= v)
        #if self.CC[cur_state] > 0:
        #    print(cur_state, a, v, mx)
        return a
    
    def receive_reward(self, reward, cur_frame, sample_achievable):
        """
        Receive r(s,a), update to s'
        """
        
        # reward = array of [success sub6, success mmWave]
        self.Success = reward
        
        total_reward = 0

        # update PLR
        for k in range(self.num_devices):
            for i in range(2):
                last_plr = self.PLR[k][i] * (cur_frame-1)
                new_plr = 0
                if self.Alloc[k][i] != 0:
                    new_plr = 1 - self.Success[k][i] / self.Alloc[k][i]
                self.PLR[k][i] = (last_plr + new_plr) / cur_frame
        #print("PLR each device: ", [sum(x)/2 for x in self.PLR])
        #print("PLR each device: ", self.PLR)
        
        # update PSR
        for k in range(self.num_devices):
            last_psr = self.PSR[k] * (cur_frame-1)
            new_psr = 1
            if sum(self.Alloc[k]) != 0:
                new_psr = sum(self.Success[k]) / sum(self.Alloc[k])
            self.PSR[k] = (last_psr + new_psr) /  cur_frame
            
            total_reward += self.PSR[k]
            #total_reward += new_psr
            
            total_reward -= 1 - int(self.PLR[k][0] <= self.PLR_req)
            total_reward -= 1 - int(self.PLR[k][1] <= self.PLR_req)
        #print("PSR each device: ", self.PSR)
        
        # update known average rate
        for k in range(self.num_devices):
            for i in range(2):
                old_rate = self.known_average_rate[k][i] * (cur_frame-1)
                new_rate = sample_achievable[k][i]
                self.known_average_rate[k][i] = (old_rate + new_rate) / cur_frame
        
        return total_reward
    
    def map_action(self, action_chosen):
        """
        Given tuple {x}^k, x in (0,1,2), map to [ <mm_i, sub6_i> ]
        """
        for i, action in enumerate(action_chosen):
            if action == 0:
                self.Alloc[i][0] = max(0, min(int(self.known_average_rate[i][0] * self.frame_duration / self.packet_size), self.Lk[i]))
                self.Alloc[i][1] = 0
            elif action == 1:
                self.Alloc[i][0] = 0
                self.Alloc[i][1] = max(0, min(int(self.known_average_rate[i][1] * self.frame_duration / self.packet_size), self.Lk[i]))
            else:
                self.Alloc[i][1] = max(0, min(int(self.known_average_rate[i][1] * self.frame_duration / self.packet_size), self.Lk[i]-1))
                self.Alloc[i][0] = max(0, min(int(self.known_average_rate[i][0] * self.frame_duration / self.packet_size), self.Lk[i] - self.Alloc[i][1]))
        return self.Alloc
    
    def get_max_action(self, Q, s):
        """
        Given Q and s, return [a, v] such that v = Q[s][a] max
        """
        if s not in Q:
            return [self.get_random_action_tuple(), 0]
        state = Q[s]
        curMaxNegAction = [None, -1e9]
        curMaxPosAction = [None, -1e9]
        negActionCount = 0
        for action in itertools.product(range(3), repeat=self.num_devices):
            if action not in state:
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
            if action not in state:
                aList.append(action)
        return [random.choice(aList), 0]
    
    def get_current_action(self):
        """
        Env ask for best action
        """
        Q_hat_chosen = random.choice(range(self.num_Qtable))
        self.exploration_rate *= self.decay_factor
        r1 = random.random()
        if r1 < self.exploration_rate:
            print("RANDOM!!!!")
            action_chosen = self.get_random_action_tuple()
        else:
            action_chosen = self.get_action_tuple(Q_hat_chosen)
            
        A = sum(k in (0, 2) for k in action_chosen)
        B = sum(k in (1, 2) for k in action_chosen)
            
        return action_chosen, (A <= self.num_sub6 and B <= self.num_mmWave)
        
    def calc_utility(self, x):
        return -math.exp(self.utility_func_param * x) 
        
    def update_to_new_state(self, reward, cur_frame, action, sample_achievable_rate):
        """
        Env send result of action a(t), calc r(t), update to s(t+1)
        """
        assert(self.cur_state)
        old_state = self.cur_state
        
        rew = self.receive_reward(reward, cur_frame, sample_achievable_rate)
        self.update_state()
        
        new_state = self.cur_state
        #self.CC[new_state] += 1
        #print("Old state: ", old_state)
        #print("New state: ", new_state)
        
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
            newQ = oldQ + oldA * (self.calc_utility(rew + self.discount_factor * max_QA - oldQ) - x0) # eq (21)
            
            self.Count[i][old_state][action] += 1 # line 14+15            
            self.Q_table[i][old_state][action] = newQ
        return rew