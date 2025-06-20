#deepqlearning_riskaverse.py

from dqn import DQNetwork

import random
import numpy as np
from collections import deque, namedtuple
import itertools
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define a named tuple for transitions stored in the replay memory
# s_orig: original state tuple from environment
# eta_idx: index of the discrete eta value for the current state (η_t)
# action_orig_idx: index of the original environment action (a_t)
# next_eta_idx: index of the discrete eta value chosen as part of the action (η_{t+1})
# reward_raw: the raw reward r_t from the environment
# next_s_orig: original next state tuple from environment (s_{t+1})
RiskTransition = namedtuple('RiskTransition', (
    's_orig_tuple', 'current_eta_idx',
    'action_orig_idx', 'chosen_next_eta_idx',
    'reward_raw', 'next_s_orig_tuple'
))

class RiskAverseDeepQLearning:
    def __init__(self, K, N, M, I_unused, Ts, d,
                 # --- Standard DQN params ---
                 replay_memory_capacity=5000,
                 batch_size=64,
                 gamma=0.9, # discount_factor
                 eps_start=0.999,
                 eps_end=0.05,
                 eps_decay=0.995,
                 target_update_freq=100, # C from image (steps)
                 learning_rate=1e-4,
                 dqn_hidden_layers=None,
                 St = 1500,
                 # --- Risk-Averse Specific Params ---
                 lambda_risk=0.5,  # λ_t in the paper (balancing expectation and CVaR)
                 alpha_cvar=0.05,   # α_t, confidence level for CVaR (e.g., 0.05 for 95% CVaR)
                 num_eta_levels=20, # D, number of discrete levels for η
                 eta_min_val=-20.0, # Lower bound for η discretization (e.g., min expected reward)
                 eta_max_val=20.0   # Upper bound for η discretization (e.g., max expected reward)
                 ):

        # Parameters from the original signature
        self.num_devices = K
        self.num_sub6 = N
        self.num_mmWave = M
        self.frame_duration = Ts
        self.packet_size = d
        self.cold_start = St
        
        # Q-learning and agent parameters
        self.exploration_rate = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_factor = eps_decay
        self.discount_factor = gamma # GAMMA

        # State and reward tracking (original parts)
        self.PLR_req = 0.1
        self.Lk = [6] * K
        self.known_average_rate =  [[self.packet_size / min(1,Ts) * self.Lk[_],self.packet_size / min(1,Ts) * self.Lk[_]] for _ in range(K)]
        self.Success = [[0, 0] for _ in range(K)]
        self.Alloc = [[0, 0] for _ in range(K)]
        self.PLR = [[0.0, 0.0] for _ in range(K)]
        self.PSR = [1.0 for _ in range(K)]
        
        self.cur_state_orig_tuple = None # Stores only the original environment state part

        # --- Risk-Averse Specific Attributes ---
        self.lambda_risk = lambda_risk
        self.alpha_cvar = alpha_cvar
        if not (0 < self.alpha_cvar <= 1):
            raise ValueError("alpha_cvar must be in (0, 1]")
        if not (0 <= self.lambda_risk <= 1):
            raise ValueError("lambda_risk must be in [0, 1]")

        self.num_eta_levels = num_eta_levels
        self.eta_min_val = eta_min_val
        self.eta_max_val = eta_max_val
        if self.num_eta_levels > 1:
            self.eta_discrete_values = torch.linspace(self.eta_min_val, self.eta_max_val, self.num_eta_levels)
        elif self.num_eta_levels == 1:
             self.eta_discrete_values = torch.tensor([(self.eta_min_val + self.eta_max_val) / 2.0])
        else:
            raise ValueError("num_eta_levels must be at least 1.")
        
        self.current_eta_idx = self.num_eta_levels // 2
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} for RiskAverseDQN")
        self.eta_discrete_values = self.eta_discrete_values.to(self.device)


        # DQN specific attributes
        # The input to DQNetwork will be: original_state_features + 1 (for current_eta_value)
        # The output of DQNetwork will be: num_original_actions * num_eta_levels
        self.num_original_actions = 3 ** self.num_devices
        
        # DQNetwork input_features = 4 * K (original state) + 1 (current eta value)
        # DQNetwork output_features = (3^K actions) * num_eta_levels (for choosing next_eta_idx)
        policy_input_features = 4 * self.num_devices + 1 
        policy_output_features = self.num_original_actions * self.num_eta_levels

        self.policy_net = DQNetwork(num_devices=self.num_devices, # This argument might be misleading for DQNetwork
                                    hidden_layer_list=dqn_hidden_layers,
                                    # Override input/output for risk-averse case
                                    _input_features_override=policy_input_features,
                                    _output_features_override=policy_output_features
                                    ).to(self.device)
        self.target_net = DQNetwork(num_devices=self.num_devices,
                                    hidden_layer_list=dqn_hidden_layers,
                                    _input_features_override=policy_input_features,
                                    _output_features_override=policy_output_features
                                    ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.replay_memory = deque(maxlen=replay_memory_capacity)
        self.BATCH_SIZE = batch_size
        self.TARGET_UPDATE_FREQUENCY = target_update_freq
        self.total_steps_done = 0

        self._original_action_tuples_list = list(itertools.product(range(3), repeat=self.num_devices))
        self._original_action_to_index_map = {action_tuple: i for i, action_tuple in enumerate(self._original_action_tuples_list)}
        self._original_index_to_action_map = {i: action_tuple for i, action_tuple in enumerate(self._original_action_tuples_list)}
        
        self.init_state() # Initializes self.cur_state_orig_tuple

    def _get_current_eta_value(self):
        return self.eta_discrete_values[self.current_eta_idx]

    def _state_to_tensor(self, state_orig_tuple, eta_value_tensor):
        """Converts original state tuple and eta_value tensor to a combined PyTorch tensor."""
        if state_orig_tuple is None: return None
        s_tensor = torch.tensor(list(state_orig_tuple), dtype=torch.float32, device=self.device)
        # Ensure eta_value_tensor is [1] or compatible for cat
        if eta_value_tensor.ndim == 0:
            eta_value_tensor = eta_value_tensor.unsqueeze(0)
        return torch.cat((s_tensor, eta_value_tensor), dim=0).unsqueeze(0) # batch dim

    # --- Functions to keep similar to DeepQLearning, but adapt for state/action ---

    def init_state(self):
        """Initialize original state part. Eta is handled by self.current_eta_idx."""
        state_list = []
        for k_idx in range(self.num_devices):
            sub6_success_init = random.choice(range(self.Lk[k_idx] + 1))
            mmwave_success_init = self.Lk[k_idx] - sub6_success_init
            state_list.extend([
                random.choice([0, 1]), random.choice([0, 1]),
                sub6_success_init, mmwave_success_init
            ])
        self.cur_state_orig_tuple = tuple(state_list)
        # Also reset current_eta_idx for a new episode/init
        self.current_eta_idx = self.num_eta_levels // 2


    def update_state(self):
        """Update original state part. Eta is updated based on action."""
        state_list = []
        for k_idx in range(self.num_devices):
            state_list.extend([
                int(self.PLR[k_idx][0] <= self.PLR_req),
                int(self.PLR[k_idx][1] <= self.PLR_req),
                self.Success[k_idx][0], self.Success[k_idx][1]
            ])
        self.cur_state_orig_tuple = tuple(state_list)

    def get_random_action_tuple(self):
        """Returns (random_original_action_tuple, random_next_eta_idx)"""
        random_orig_action_tuple = tuple(random.choices(range(3), k=self.num_devices))
        random_next_eta_idx = random.randrange(self.num_eta_levels)
        return random_orig_action_tuple, random_next_eta_idx

    def get_action_tuple(self, current_full_state_tensor):
        """
        Selects (original_action_tuple, next_eta_idx) using the policy network.
        Args:
            current_full_state_tensor (torch.Tensor): Combined (original_state, current_eta_value).
        Returns:
            tuple: (chosen_original_action_tuple, chosen_next_eta_idx)
        """
        with torch.no_grad():
            q_values_all_composite_actions = self.policy_net(current_full_state_tensor) # Shape [1, num_orig_actions * num_eta_levels]
            
            # Find the index of the max Q-value (this is a flat index)
            composite_action_flat_idx = q_values_all_composite_actions.argmax(dim=1).item()
            
            # Convert flat index back to (original_action_idx, next_eta_idx)
            chosen_original_action_idx = composite_action_flat_idx // self.num_eta_levels
            chosen_next_eta_idx = composite_action_flat_idx % self.num_eta_levels
            
            chosen_original_action_tuple = self._original_index_to_action_map[chosen_original_action_idx]
        return chosen_original_action_tuple, chosen_next_eta_idx

    # receive_reward and map_action remain identical to DeepQLearning as they deal with original rewards/actions
    def receive_reward(self, env_reward_signal, current_frame_number, sample_achievable_rates):
        self.Success = env_reward_signal
        total_scalar_raw_reward = 0.0
        for k_idx in range(self.num_devices):
            for band_idx in range(2):
                sum_past_plr = self.PLR[k_idx][band_idx] * (current_frame_number - 1)
                current_plr_value = 0.0
                if self.Alloc[k_idx][band_idx] > 0:
                    current_plr_value = 1.0 - (self.Success[k_idx][band_idx] / self.Alloc[k_idx][band_idx])
                self.PLR[k_idx][band_idx] = (sum_past_plr + current_plr_value) / current_frame_number
        for k_idx in range(self.num_devices):
            sum_past_psr = self.PSR[k_idx] * (current_frame_number - 1)
            current_psr_value = 1.0
            if sum(self.Alloc[k_idx]) > 0:
                current_psr_value = sum(self.Success[k_idx]) / sum(self.Alloc[k_idx])
            self.PSR[k_idx] = (sum_past_psr + current_psr_value) / current_frame_number
            total_scalar_raw_reward += current_psr_value
            total_scalar_raw_reward -= (1 - int(self.PLR[k_idx][0] <= self.PLR_req))
            total_scalar_raw_reward -= (1 - int(self.PLR[k_idx][1] <= self.PLR_req))
        for k_idx in range(self.num_devices):
            for band_idx in range(2):
                A = 0.7
                sum_past_rates = self.known_average_rate[k_idx][band_idx] * A
                current_rate_sample = sample_achievable_rates[k_idx][band_idx] * (1.0-A)
                self.known_average_rate[k_idx][band_idx] = (sum_past_rates + current_rate_sample)
        return total_scalar_raw_reward


    def map_action(self, original_action_chosen_tuple):
        # Uses the original action part to determine allocations
        for dev_idx, action_type in enumerate(original_action_chosen_tuple):
            dev_lk = self.Lk[dev_idx]
            est_packets_sub6 = int(self.known_average_rate[dev_idx][0] * self.frame_duration / self.packet_size)
            est_packets_mmwave = int(self.known_average_rate[dev_idx][1] * self.frame_duration / self.packet_size)
            if action_type == 0:
                self.Alloc[dev_idx][0] = max(0, min(est_packets_sub6, dev_lk))
                self.Alloc[dev_idx][1] = 0
            elif action_type == 1:
                self.Alloc[dev_idx][0] = 0
                self.Alloc[dev_idx][1] = max(0, min(est_packets_mmwave, dev_lk))
            else:
                alloc_mmwave_limit = dev_lk - 1 if dev_lk > 0 else 0
                self.Alloc[dev_idx][1] = max(0, min(est_packets_mmwave, alloc_mmwave_limit))
                remaining_capacity_for_sub6 = dev_lk - self.Alloc[dev_idx][1]
                self.Alloc[dev_idx][0] = max(0, min(est_packets_sub6, remaining_capacity_for_sub6))
        return self.Alloc

    def get_current_action(self, cur_frame=0):
        """Selects (original_action_tuple, next_eta_idx) using epsilon-greedy."""
        #self.exploration_rate = self.eps_end + \
        #                        (self.eps_start - self.eps_end) * \
        #                        math.exp(-1. * self.total_steps_done * (1./(1/self.decay_factor)) )
        if cur_frame >= self.cold_start:
            if self.exploration_rate >= self.eps_end:
                self.exploration_rate *= self.decay_factor
        
        rand_sample = random.random()
        if rand_sample < self.exploration_rate:
            chosen_orig_action_tuple, chosen_next_eta_idx = self.get_random_action_tuple()
        else:
            current_eta_val_tensor = self._get_current_eta_value().detach() # Ensure it's a scalar tensor
            current_full_state_tensor = self._state_to_tensor(self.cur_state_orig_tuple, current_eta_val_tensor)
            chosen_orig_action_tuple, chosen_next_eta_idx = self.get_action_tuple(current_full_state_tensor)
            
        # Constraint check based on original action part
        num_active_sub6 = sum(1 for k_act_type in chosen_orig_action_tuple if k_act_type in (0, 2))
        num_active_mmwave = sum(1 for k_act_type in chosen_orig_action_tuple if k_act_type in (1, 2))
        constraints_met = (num_active_sub6 <= self.num_sub6 and num_active_mmwave <= self.num_mmWave)
        
        return chosen_orig_action_tuple, chosen_next_eta_idx, constraints_met

    def _optimize_model(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return None

        transitions = random.sample(self.replay_memory, self.BATCH_SIZE)
        batch = RiskTransition(*zip(*transitions))

        # Unpack batch
        s_orig_batch_tensor = torch.tensor(batch.s_orig_tuple, dtype=torch.float32, device=self.device)
        current_eta_idx_batch = torch.tensor(batch.current_eta_idx, device=self.device, dtype=torch.long)
        action_orig_idx_batch = torch.tensor(batch.action_orig_idx, device=self.device, dtype=torch.long)
        chosen_next_eta_idx_batch = torch.tensor(batch.chosen_next_eta_idx, device=self.device, dtype=torch.long)
        reward_raw_batch = torch.tensor(batch.reward_raw, device=self.device, dtype=torch.float32)
        next_s_orig_batch_tensor = torch.tensor(batch.next_s_orig_tuple, dtype=torch.float32, device=self.device)
        
        # Get current eta values (η_j) from the discrete values tensor
        current_eta_val_batch = self.eta_discrete_values[current_eta_idx_batch] # Shape [BATCH_SIZE]
        
        # Get next eta values chosen as action (η_{j+1})
        chosen_next_eta_val_batch = self.eta_discrete_values[chosen_next_eta_idx_batch] # Shape [BATCH_SIZE]

        # Prepare policy_net input: Concatenate original state tensor and current eta value
        # Ensure eta_values have an extra dimension for concatenation if needed
        policy_net_input_batch = torch.cat(
            (s_orig_batch_tensor, current_eta_val_batch.unsqueeze(1)), dim=1
        ) # Shape [BATCH_SIZE, num_orig_features + 1]

        # Q( (s_j, η_j), (a_j, η_{j+1}) )
        # Q_values for all (orig_action, next_eta) pairs for each state in batch
        all_q_for_current_state = self.policy_net(policy_net_input_batch) # Shape [BATCH_SIZE, num_orig_actions * num_eta_levels]
        
        # Construct composite action index for gathering
        composite_action_indices = action_orig_idx_batch * self.num_eta_levels + chosen_next_eta_idx_batch
        current_q_values = all_q_for_current_state.gather(1, composite_action_indices.unsqueeze(1)).squeeze(1) # Shape [BATCH_SIZE]

        # 1. Risk-adjusted immediate reward part: -(λ/α)[η_j - r_j]_+ + (1-λ)r_j
        term1_cvar_part = -(self.lambda_risk / self.alpha_cvar) * F.relu(current_eta_val_batch - reward_raw_batch)
        term1_exp_part = (1 - self.lambda_risk) * reward_raw_batch
        risk_adjusted_immediate_reward = term1_cvar_part + term1_exp_part # Shape [BATCH_SIZE]

        # 2. Middle term: γ * λ * η_{j+1}
        term2_gamma_lambda_eta_next = self.discount_factor * self.lambda_risk * chosen_next_eta_val_batch # Shape [BATCH_SIZE]
        
        # 3. Future part: γ * max_{a', η''} Q_target(s_{j+1}, η_{j+1}, a', η''; θ_target)
        target_net_input_batch = torch.cat(
            (next_s_orig_batch_tensor, chosen_next_eta_val_batch.unsqueeze(1)), dim=1
        ) # Shape [BATCH_SIZE, num_orig_features + 1]
        
        with torch.no_grad():
            q_target_all_composite_actions = self.target_net(target_net_input_batch) # Shape [BATCH_SIZE, num_orig_actions * num_eta_levels]
            max_q_target_next = q_target_all_composite_actions.max(dim=1)[0] # Max over all (a', η'') pairs. Shape [BATCH_SIZE]
        
        term3_gamma_max_q_target = self.discount_factor * max_q_target_next # Shape [BATCH_SIZE]
        
        # Combine for y_j
        expected_q_values = risk_adjusted_immediate_reward + term2_gamma_lambda_eta_next + term3_gamma_max_q_target

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()

    def update_to_new_state(self, env_reward_signal, current_frame_number,
                            action_taken_orig_tuple, chosen_next_eta_idx, # Action now has two parts
                            sample_achievable_rates):
        
        old_original_state_tuple = self.cur_state_orig_tuple
        previous_current_eta_idx = self.current_eta_idx # This is η_t

        # Calculate raw reward and update original state metrics (PLR, PSR etc.)
        raw_scalar_reward = self.receive_reward(env_reward_signal, current_frame_number, sample_achievable_rates)
        
        # Update original part of the state (s_t -> s_{t+1})
        self.update_state()
        new_original_state_tuple = self.cur_state_orig_tuple
        
        # Store transition in replay memory
        # (s_orig_t, η_t_idx, a_orig_t_idx, η_{t+1}_idx, r_raw_t, s_orig_{t+1})
        orig_action_idx = self._original_action_to_index_map[action_taken_orig_tuple]
        
        self.replay_memory.append(RiskTransition(
            old_original_state_tuple, previous_current_eta_idx,
            orig_action_idx, chosen_next_eta_idx,
            raw_scalar_reward, new_original_state_tuple
        ))

        self.current_eta_idx = chosen_next_eta_idx
        
        self.total_steps_done += 1
        loss_value = self._optimize_model()

        if self.total_steps_done % self.TARGET_UPDATE_FREQUENCY == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # if loss_value is not None:
            #    print(f"Step {self.total_steps_done}: Target net updated. RiskDQN Loss: {loss_value:.4f}, Exp: {self.exploration_rate:.3f}")


        return raw_scalar_reward # Return raw reward for external monitoring