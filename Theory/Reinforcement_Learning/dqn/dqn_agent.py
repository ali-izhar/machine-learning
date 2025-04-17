import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
from models import DQN, Transition


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        device,
        memory,
        batch_size=128,
        gamma=0.99,  # Discount factor for future rewards
        eps_start=0.9,  # Starting exploration rate
        eps_end=0.05,  # Minimum exploration rate
        eps_decay=200,  # Exploration decay rate
        tau=0.005,  # Soft update parameter for target network
    ):
        """Initialize DQN Agent

        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            device: Device to run calculations on (cpu/cuda)
            memory: Replay memory buffer
            batch_size: Minibatch size for training
            gamma: Discount factor for future rewards (γ in Bellman equation)
            eps_start: Starting epsilon for exploration
            eps_end: Minimum epsilon value
            eps_decay: Decay rate for epsilon
            tau: Rate for soft target network update (τ)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau

        # Initialize two networks: policy_net for action selection and optimization
        # and target_net for generating target Q-values to stabilize training
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is used for inference only, no gradients

        # Adam optimizer with weight decay (L2 regularization)
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=1e-4, amsgrad=True
        )

        # Steps counter for epsilon annealing calculation
        self.steps_done = 0

    def select_action(self, state):
        """Select action using epsilon-greedy policy

        Balances exploration (random actions) vs exploitation (best known action)
        by using a decaying epsilon probability for random actions.
        """
        sample = random.random()
        # Exponential decay of epsilon from eps_start to eps_end
        # Formula: ε = εend + (εstart - εend) * exp(-steps / decay_rate)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # Exploit: select action with highest Q-value
                # max(1) selects the maximum Q-value along dimension 1 (actions)
                # Returns tuple (max_values, argmax), we want the action index (argmax)
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Explore: select random action
            return torch.tensor(
                [[random.randrange(self.action_size)]],
                device=self.device,
                dtype=torch.long,
            )

    def optimize_model(self):
        """Perform one step of optimization on the DQN

        Implements the DQN loss function:
        L = (r + γ * max_a' Q_target(s', a') - Q(s, a))²

        For terminal states, the target is just the reward.
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        transitions = self.memory.sample(self.batch_size)
        # Transpose batch into separate components
        # This converts [(s1,a1,s1',r1), (s2,a2,s2',r2), ...]
        # to ([s1,s2,...], [a1,a2,...], [s1',s2',...], [r1,r2,...])
        batch = Transition(*zip(*transitions))

        # Create mask for non-final states (where next_state is not None)
        # For terminal states, future reward (Q-value) is 0
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        # Prepare batch for network
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a_t) for actions taken
        # gather(1, action_batch) selects the Q-values for the actions actually taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) = max_a Q(s_{t+1}, a) for all next states
        # Uses the target network to improve stability with "bootstrapping"
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():  # No need for gradients when computing targets
            # For non-terminal states, compute max_a Q(s', a)
            # For terminal states, next_state_values remains 0
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1)[0]

        # Compute expected Q values: γ*max_a Q(s', a) + reward
        # This is the target from the Bellman equation: r + γ * max_a Q(s', a)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Calculate loss (Huber loss provides more stability than MSE for outliers)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update target network
        # θ′ ← τ θ + (1 −τ )θ′
        # This gradually shifts target_net weights toward policy_net weights
        # Parameter τ controls the update rate (typically small, e.g., 0.005)
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
