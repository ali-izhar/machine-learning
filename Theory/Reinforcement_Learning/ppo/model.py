"""Model for PPO (Proximal Policy Optimization)"""

import torch.nn as nn
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator


def create_actor_network(state_size, action_size, num_cells=256, device="cpu"):
    """Creates a policy network for continuous action spaces.

    Args:
        state_size: Dimension of the state/observation space
        action_size: Dimension of the action space
        num_cells: Number of hidden units in each layer
        device: Device to place the network on

    Returns:
        A policy module that outputs action distribution parameters
    """
    # Actor network outputs mean and log std of a Normal distribution
    actor_net = nn.Sequential(
        nn.Linear(state_size, num_cells, device=device),
        nn.Tanh(),
        nn.Linear(num_cells, num_cells, device=device),
        nn.Tanh(),
        nn.Linear(num_cells, num_cells, device=device),
        nn.Tanh(),
        nn.Linear(num_cells, 2 * action_size, device=device),
        NormalParamExtractor(),
    )

    return actor_net


def create_critic_network(state_size, num_cells=256, device="cpu"):
    """Creates a value network (critic) for PPO.

    Args:
        state_size: Dimension of the state/observation space
        num_cells: Number of hidden units in each layer
        device: Device to place the network on

    Returns:
        A value network that outputs state value estimates
    """
    value_net = nn.Sequential(
        nn.Linear(state_size, num_cells, device=device),
        nn.Tanh(),
        nn.Linear(num_cells, num_cells, device=device),
        nn.Tanh(),
        nn.Linear(num_cells, num_cells, device=device),
        nn.Tanh(),
        nn.Linear(num_cells, 1, device=device),
    )

    return value_net


def create_policy_module(actor_net, action_spec):
    """Wraps the actor network into a TensorDictModule and ProbabilisticActor.

    Args:
        actor_net: Neural network that outputs distribution parameters
        action_spec: Action space specification from the environment

    Returns:
        A policy module that can be used for sampling actions
    """
    # Create TensorDictModule to interface with tensordict data structures
    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )

    # Create probabilistic actor that builds a TanhNormal distribution
    # and samples from it
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        return_log_prob=True,  # needed for PPO's importance sampling
    )

    return policy_module


def create_value_module(value_net):
    """Wraps the critic network into a ValueOperator.

    Args:
        value_net: Neural network that outputs value estimates

    Returns:
        A value module that can estimate state values
    """
    # Create value operator to interface with tensordict data structures
    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    return value_module
