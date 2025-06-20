# deepqlearing_simple.py

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
Transition = namedtuple('Transition', ('state', 'action_index', 'next_state', 'reward'))

class DeepQLearning:
    def __init__(self, K, N, M, I, Ts, d,
                 replay_memory_capacity=500,
                 batch_size=64,
                 gamma=0.9, # discount_factor
                 eps_start=0.999, # exploration_rate start
                 eps_end=0.05,
                 eps_decay=0.995, # decay_factor for exploration_rate
                 target_update_freq=100, # C from image (steps)
                 learning_rate=1e-4,
                 dqn_hidden_layers=None, # Pass to DQNetwork, e.g., [128, 64]
                 St = 1500,
                 ):
        # Parameters from the original signature
        self.num_devices = K
        self.num_sub6 = N      # Max devices on sub6, used in get_current_action constraint check
        self.num_mmWave = M    # Max devices on mmWave, used in get_current_action constraint check
        # I (num_Qtable from RiskAverseQLearning) is unused but kept for signature compatibility
        self.frame_duration = Ts
        self.packet_size = d
        self.cold_start = St
        
        # Q-learning and agent parameters
        self.exploration_rate = eps_start # Current epsilon for epsilon-greedy
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_factor = eps_decay # Multiplicative decay factor for exploration_rate
        self.discount_factor = gamma  # GAMMA for Q-learning updates

        # State and reward tracking, same as RiskAverseQLearning
        self.PLR_req = 0.1
        self.Lk = [6] * K # Example: Max packets per device
        
        self.known_average_rate = [[self.packet_size / min(1,Ts) * self.Lk[_],self.packet_size / min(1,Ts) * self.Lk[_]] for _ in range(K)] # [sub6, mmWave]
        self.Success = [[0, 0] for _ in range(K)] # [sub6, mmWave] successful packets
        self.Alloc = [[0, 0] for _ in range(K)]   # [sub6, mmWave] allocated packets
        self.PLR = [[0.0, 0.0] for _ in range(K)] # [sub6, mmWave] Packet Loss Rate
        self.PSR = [1.0 for _ in range(K)]        # Overall Packet Success Rate per device
        
        self.cur_state = None # This will be a Python tuple representing the current state
        self.init_state()     # Initialize self.cur_state

        # DQN specific attributes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize Policy Network and Target Network
        self.policy_net = DQNetwork(num_devices=K, hidden_layer_list=dqn_hidden_layers).to(self.device)
        self.target_net = DQNetwork(num_devices=K, hidden_layer_list=dqn_hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only for inference, not training directly

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.replay_memory = deque(maxlen=replay_memory_capacity)
        self.BATCH_SIZE = batch_size
        self.TARGET_UPDATE_FREQUENCY = target_update_freq # C: steps to update target network
        
        self.total_steps_done = 0 # For decaying epsilon and updating target network

        # Helper for mapping actions (tuples) to indices and vice-versa
        # Action: tuple of length K, each element in {0, 1, 2}
        # 0: sub6 only, 1: mmWave only, 2: both
        self._action_tuples_list = list(itertools.product(range(3), repeat=self.num_devices))
        self._action_to_index_map = {action_tuple: i for i, action_tuple in enumerate(self._action_tuples_list)}
        self._index_to_action_map = {i: action_tuple for i, action_tuple in enumerate(self._action_tuples_list)}

    def _state_to_tensor(self, state_tuple):
        """Converts a state tuple to a PyTorch tensor for network input."""
        if state_tuple is None:
            return None
        # Ensure the state is flat list of numbers before converting to tensor
        return torch.tensor(list(state_tuple), dtype=torch.float32, device=self.device).unsqueeze(0)

    def init_state(self):
        """Initialize to a random state (tuple). Kept from RiskAverseQLearning."""
        state_list = []
        for k_idx in range(self.num_devices):
            # Example state components:
            # - PLR requirement met for sub6 (0 or 1)
            # - PLR requirement met for mmWave (0 or 1)
            # - Number of successful transmissions on sub6 in last step
            # - Number of successful transmissions on mmWave in last step
            # For init, we can use random values or typical starting values.
            # The original used random choices for 'a' and 'b' for success counts.
            # Let's use Lk[k_idx] as a reference for success counts init.
            # Assuming Success components are counts, not rates for state.
            sub6_success_init = random.choice(range(self.Lk[k_idx] + 1))
            mmwave_success_init = self.Lk[k_idx] - sub6_success_init # Ensure sum is Lk for this example part
            
            state_list.extend([
                random.choice([0, 1]),  # Mock PLR sub6 met
                random.choice([0, 1]),  # Mock PLR mmWave met
                sub6_success_init,      # Mock last success sub6 (count)
                mmwave_success_init     # Mock last success mmWave (count)
            ])
        self.cur_state = tuple(state_list)

    def update_state(self):
        """Update the current state (tuple) based on PLR and Success. Kept from RiskAverseQLearning."""
        state_list = []
        for k_idx in range(self.num_devices):
            state_list.extend([
                int(self.PLR[k_idx][0] <= self.PLR_req),  # PLR sub6 requirement met
                int(self.PLR[k_idx][1] <= self.PLR_req),  # PLR mmWave requirement met
                self.Success[k_idx][0],                   # Actual successful packets sub6
                self.Success[k_idx][1]                    # Actual successful packets mmWave
            ])
        self.cur_state = tuple(state_list)

    def get_random_action_tuple(self):
        """Get a random action tuple. Kept from RiskAverseQLearning."""
        return tuple(random.choices(range(3), k=self.num_devices))

    def get_action_tuple(self, state_tensor_for_net):
        """
        Selects an action using the policy network based on the current state tensor.
        Args:
            state_tensor_for_net (torch.Tensor): The current state as a tensor [1, num_features].
        Returns:
            tuple: The action tuple selected by the policy network.
        """
        with torch.no_grad(): # No gradient needed for action selection
            # policy_net outputs Q-values for all 3^K actions
            q_values = self.policy_net(state_tensor_for_net)
            # Select action with the highest Q-value
            action_index = q_values.max(1)[1].item() # .max(1) returns (values, indices)
        return self._index_to_action_map[action_index]

    def receive_reward(self, env_reward_signal, current_frame_number, sample_achievable_rates):
        """
        Calculate scalar reward based on environment feedback and update internal metrics (PLR, PSR, known_average_rate).
        Kept similar to RiskAverseQLearning.
        Args:
            env_reward_signal (list of lists): [[success_sub6_dev0, success_mmwave_dev0], ...]
            current_frame_number (int): The current frame number (1-indexed).
            sample_achievable_rates (list of lists): [[rate_sub6_dev0, rate_mmwave_dev0], ...]
        Returns:
            float: The calculated scalar reward.
        """
        self.Success = env_reward_signal # Update success counts based on environment feedback
        
        total_scalar_reward = 0.0

        # Update PLR for each device and each band
        for k_idx in range(self.num_devices):
            for band_idx in range(2): # 0 for sub6, 1 for mmWave
                # Calculate sum of past PLR values to maintain running average
                # current_frame_number is 1-indexed. For frame 1, (current_frame_number-1) is 0.
                sum_past_plr = self.PLR[k_idx][band_idx] * (current_frame_number - 1)
                
                current_plr_value = 0.0
                if self.Alloc[k_idx][band_idx] > 0:
                    current_plr_value = 1.0 - (self.Success[k_idx][band_idx] / self.Alloc[k_idx][band_idx])
                # If Alloc is 0, PLR is 0 (no packets sent, so no packets lost)
                
                self.PLR[k_idx][band_idx] = (sum_past_plr + current_plr_value) / current_frame_number
        
        # Update PSR (Packet Success Rate) and calculate part of the reward
        for k_idx in range(self.num_devices):
            sum_past_psr = self.PSR[k_idx] * (current_frame_number - 1)
            
            current_psr_value = 1.0 # Default PSR is 1 (e.g. if no packets allocated)
            if sum(self.Alloc[k_idx]) > 0:
                current_psr_value = sum(self.Success[k_idx]) / sum(self.Alloc[k_idx])
            
            self.PSR[k_idx] = (sum_past_psr + current_psr_value) / current_frame_number
            
            total_scalar_reward += current_psr_value
            
            # Penalize if PLR requirements are not met
            total_scalar_reward -= (1 - int(self.PLR[k_idx][0] <= self.PLR_req)) # Penalty for sub6 PLR miss
            total_scalar_reward -= (1 - int(self.PLR[k_idx][1] <= self.PLR_req)) # Penalty for mmWave PLR miss
        
        # Update known average achievable rate
        for k_idx in range(self.num_devices):
            for band_idx in range(2):
                A = 0.7
                sum_past_rates = self.known_average_rate[k_idx][band_idx] * A
                current_rate_sample = sample_achievable_rates[k_idx][band_idx] * (1.0-A)
                self.known_average_rate[k_idx][band_idx] = (sum_past_rates + current_rate_sample)
        
        return total_scalar_reward

    def map_action(self, action_chosen_tuple):
        """
        Maps the chosen action tuple to resource allocations (self.Alloc).
        Kept from RiskAverseQLearning.
        Args:
            action_chosen_tuple (tuple): The action chosen, e.g., (0, 1, 2) for K devices.
        Returns:
            list of lists: The updated self.Alloc.
        """
        for dev_idx, action_type in enumerate(action_chosen_tuple):
            dev_lk = self.Lk[dev_idx] # Max packets for this device
            # Estimated packets based on known average rate, frame duration, and packet size
            est_packets_sub6 = int(self.known_average_rate[dev_idx][0] * self.frame_duration / self.packet_size)
            est_packets_mmwave = int(self.known_average_rate[dev_idx][1] * self.frame_duration / self.packet_size)

            if action_type == 0: # Sub6 only
                self.Alloc[dev_idx][0] = max(0, min(est_packets_sub6, dev_lk))
                self.Alloc[dev_idx][1] = 0
            elif action_type == 1: # mmWave only
                self.Alloc[dev_idx][0] = 0
                self.Alloc[dev_idx][1] = max(0, min(est_packets_mmwave, dev_lk))
            else: # action_type == 2, Both (prioritize mmWave up to Lk-1, then sub6)
                # Allocate to mmWave, reserving at least 1 slot for sub6 if Lk > 0
                alloc_mmwave_limit = dev_lk - 1 if dev_lk > 0 else 0
                self.Alloc[dev_idx][1] = max(0, min(est_packets_mmwave, alloc_mmwave_limit))
                
                remaining_capacity_for_sub6 = dev_lk - self.Alloc[dev_idx][1]
                self.Alloc[dev_idx][0] = max(0, min(est_packets_sub6, remaining_capacity_for_sub6))
        return self.Alloc

    def get_current_action(self, cur_frame):
        """
        Selects an action using epsilon-greedy strategy:
        With probability epsilon, selects a random action.
        Otherwise, selects the best action according to the policy network.
        Also returns a flag indicating if the action respects resource constraints.
        """
        # Decay exploration rate
        #self.exploration_rate = self.eps_end + \
        #                        (self.eps_start - self.eps_end) * \
        #                        math.exp(-1. * self.total_steps_done * (1./(1/self.decay_factor)) ) # Using typical exponential decay related to decay_factor interpreation

        # Alternative simpler multiplicative decay:
        if cur_frame >= self.cold_start:
            if self.exploration_rate > self.eps_end:
                self.exploration_rate *= self.decay_factor


        rand_sample = random.random()
        if rand_sample < self.exploration_rate:
            action_chosen_tuple = self.get_random_action_tuple()
            # print(f"Step {self.total_steps_done}: RANDOM action! Eps: {self.exploration_rate:.3f}") # For debugging
        else:
            current_state_tensor = self._state_to_tensor(self.cur_state)
            action_chosen_tuple = self.get_action_tuple(current_state_tensor)
            # print(f"Step {self.total_steps_done}: DQN action. Eps: {self.exploration_rate:.3f}") # For debugging
            
        # Check constraints (A: num devices on sub6, B: num devices on mmWave)
        # Action type 0 (sub6), 1 (mmWave), 2 (both)
        num_active_sub6 = sum(1 for k_act_type in action_chosen_tuple if k_act_type in (0, 2))
        num_active_mmwave = sum(1 for k_act_type in action_chosen_tuple if k_act_type in (1, 2))
            
        constraints_met = (num_active_sub6 <= self.num_sub6 and num_active_mmwave <= self.num_mmWave)
        
        return action_chosen_tuple, constraints_met

    def _optimize_model(self):
        """Performs a single step of optimization on the policy network using a batch from replay memory."""
        if len(self.replay_memory) < self.BATCH_SIZE:
            return None # Not enough samples in memory to form a batch

        # Sample a random minibatch of transitions from the replay memory
        transitions = random.sample(self.replay_memory, self.BATCH_SIZE)
        # Convert batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Concatenate batch elements for PyTorch processing
        # batch.state, batch.next_state are tuples of tensors ([1, num_features])
        state_batch = torch.cat(batch.state)
        # batch.action_index is a tuple of integer action indices
        action_batch = torch.tensor(batch.action_index, device=self.device, dtype=torch.long).unsqueeze(1)
        # batch.reward is a tuple of scalar reward tensors ([1])
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        
        # Compute Q(s_t, a_t)
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) = max_{a'} Q_target(s_{t+1}, a')
        with torch.no_grad():
            next_state_max_q_values = self.target_net(next_state_batch).max(1)[0]
        
        # Compute the expected Q-values (y_j = r_j + gamma * V(s_{t+1}))
        expected_q_values = reward_batch + (self.discount_factor * next_state_max_q_values)

        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        # loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(1))

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item() # Return loss value for monitoring

    def update_to_new_state(self, env_reward_signal, current_frame_number, action_taken_tuple, sample_achievable_rates):
        """
        Processes the outcome of an action:
        1. Calculates the scalar reward and updates internal state metrics.
        2. Updates the agent's current state (s_t -> s_{t+1}).
        3. Stores the transition (s_t, a_t, r_t, s_{t+1}) in replay memory.
        4. Performs a model optimization step (training).
        5. Periodically updates the target network.
        Args:
            env_reward_signal (list of lists): Feedback from env (e.g., success counts).
            current_frame_number (int): Current frame/timestep number.
            action_taken_tuple (tuple): The action that was executed.
            sample_achievable_rates (list of lists): Observed achievable rates.
        Returns:
            float: The scalar reward received for the transition.
        """
        old_state_tuple = self.cur_state # s_t (Python tuple)
        old_state_tensor = self._state_to_tensor(old_state_tuple) # Convert to tensor

        # Calculate scalar reward (r_t) and update internal metrics based on env_reward_signal
        actual_scalar_reward = self.receive_reward(env_reward_signal, current_frame_number, sample_achievable_rates)
        reward_tensor = torch.tensor([actual_scalar_reward], device=self.device, dtype=torch.float32)
        
        # Update current state to new_state (s_{t+1}) based on updated metrics
        self.update_state()
        new_state_tuple = self.cur_state # s_{t+1} (Python tuple)
        new_state_tensor = self._state_to_tensor(new_state_tuple) # Convert to tensor
        
        # Convert action_taken_tuple to its index for storage
        action_index = self._action_to_index_map[action_taken_tuple]
        
        # Store the transition in replay memory D
        self.replay_memory.append(Transition(old_state_tensor, action_index, new_state_tensor, reward_tensor))

        # Increment step counter
        self.total_steps_done += 1

        # Perform one step of the optimization (on the policy network)
        loss_value = self._optimize_model() # This will only run if memory has enough samples

        # Periodically update the target network with weights from the policy network (every C steps)
        if self.total_steps_done % self.TARGET_UPDATE_FREQUENCY == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # print(f"--- Step {self.total_steps_done}: Target network updated. Exploration: {self.exploration_rate:.4f} ---")
            # if loss_value is not None:
            #    print(f"Training Loss: {loss_value:.4f}")

        return actual_scalar_reward