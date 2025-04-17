import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple, deque

# Define transition tuple structure for experience replay
# Each experience is stored as (s, a, s', r) where:
# - s: state
# - a: action taken
# - s': next state (or None if terminal)
# - r: reward received
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        """Initialize a fixed-size memory buffer

        Experience replay breaks the correlation between consecutive samples
        by storing and randomly sampling experiences, making training more stable.
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a random batch from memory

        Random sampling breaks temporal correlations in the training data,
        improving stability of neural network training.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        """Q-Network architecture

        Maps state observations to action-values (Q-values) for each possible action.
        Q(s,a) represents the expected future reward of taking action a in state s.
        """
        super(DQN, self).__init__()
        # Two hidden layers of 128 units each with ReLU activation
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        # Output layer produces Q-values for each possible action
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """Forward pass computes Q(s,a) for all actions given state s"""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # No activation on output layer - Q-values can be positive or negative
        return self.layer3(x)
