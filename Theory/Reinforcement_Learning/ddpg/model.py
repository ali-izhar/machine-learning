"""Model for Deep Deterministic Policy Gradient (DDPG)"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    # Calculates initialization range based on fan-in (number of input units)
    # Using the formula from the DDPG paper: initialize weights uniformly in range [-1/sqrt(f), 1/sqrt(f)]
    # where f is the fan-in of the layer
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Neural network architecture follows the DDPG paper recommendations:
        # Hidden layers with 400 and 300 units using ReLU activations
        self.fc1 = nn.Linear(state_size, fc1_units)  # First fully connected layer
        self.fc2 = nn.Linear(fc1_units, fc2_units)  # Second fully connected layer
        self.fc3 = nn.Linear(
            fc2_units, action_size
        )  # Output layer mapping to action space
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize parameters with proper scaling for better convergence
        # The final layer is initialized with smaller weights to ensure initial outputs are near zero
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # Final layer initialized with smaller values to ensure initial random policies
        # have small initial magnitude
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        # Forward pass implementing the deterministic policy μ(s|θ^μ)
        x = F.relu(self.fc1(state))  # Apply ReLU activation to first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation to second layer
        # Output layer uses tanh to bound actions to [-1, 1]
        # This is common in continuous control as actions are often normalized
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Architecture for Q(s,a|θ^Q) function approximation
        # First processes the state input
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        # Then combines state features with action and processes through second layer
        # This architecture allows actions to affect the network at a deeper layer,
        # giving states more influence on the network's early representations
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        # Output layer produces a single Q-value
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        # Similar initialization strategy as Actor, with proper scaling for each layer
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # Final layer initialized with smaller values for better initial approximation
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Forward pass implementing the Q-function Q(s,a|θ^Q)
        # First, process the state input
        xs = F.relu(self.fcs1(state))
        # Concatenate state features with the action (architecture from DDPG paper)
        # This allows the action to influence the network at the second layer
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        # Output is the predicted Q-value (no activation function on output)
        # Q-values are unbounded, unlike actions which are typically bounded
        return self.fc3(x)
