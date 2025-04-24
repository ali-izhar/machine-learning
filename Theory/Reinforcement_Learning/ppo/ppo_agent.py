"""PPO Agent"""

import torch
from collections import defaultdict

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.collectors import SyncDataCollector
from torchrl.envs.utils import set_exploration_type, ExplorationType

from model import (
    create_actor_network,
    create_critic_network,
    create_policy_module,
    create_value_module,
)

# Default PPO hyperparameters
PPO_CLIP_EPSILON = 0.2
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_EPS = 1e-4
LR = 3e-4
MAX_GRAD_NORM = 1.0
NUM_EPOCHS = 10
SUB_BATCH_SIZE = 64


class PPOAgent:
    """Proximal Policy Optimization agent implementation.

    PPO is an on-policy algorithm that uses:
    - Actor-critic architecture
    - Clipped surrogate objective to constrain policy updates
    - Generalized Advantage Estimation (GAE)
    - Entropy bonus for exploration
    """

    def __init__(
        self,
        env,
        device=None,
        num_cells=256,
        clip_epsilon=PPO_CLIP_EPSILON,
        gamma=GAMMA,
        lmbda=LAMBDA,
        entropy_eps=ENTROPY_EPS,
        lr=LR,
        max_grad_norm=MAX_GRAD_NORM,
        loss_critic_type="smooth_l1",
    ):
        """
        Initialize a PPO Agent.

        Args:
            env: The environment to interact with
            device: Device to place tensors on ('cpu' or 'cuda')
            num_cells: Number of cells in hidden layers
            clip_epsilon: Clipping parameter for PPO loss
            gamma: Discount factor
            lmbda: GAE lambda parameter for advantage estimation
            entropy_eps: Entropy bonus coefficient
            lr: Learning rate
            max_grad_norm: Maximum norm for gradient clipping
            loss_critic_type: Type of value loss function ('mse' or 'smooth_l1')
        """
        self.env = env
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Get environment specs
        state_size = env.observation_spec["observation"].shape[-1]
        self.action_size = env.action_spec.shape[-1]
        self.action_spec = env.action_spec

        # Create networks
        self.actor_net = create_actor_network(
            state_size, self.action_size, num_cells, device=self.device
        )
        self.critic_net = create_critic_network(
            state_size, num_cells, device=self.device
        )

        # Create modules
        self.policy = create_policy_module(self.actor_net, self.action_spec)
        self.value = create_value_module(self.critic_net)

        # Initialize advantage estimation module
        self.advantage_module = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=self.value,
            average_gae=True,
            device=self.device,
        )

        # Initialize PPO loss module
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.value,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_eps),
            entropy_coef=entropy_eps,
            critic_coef=1.0,
            loss_critic_type=loss_critic_type,
        )

        # Set up optimizer
        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), lr=lr)
        self.max_grad_norm = max_grad_norm

        # Store hyperparameters
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps

    def create_collector(self, frames_per_batch, total_frames):
        """Create a data collector for gathering experience.

        Args:
            frames_per_batch: Number of frames to collect per batch
            total_frames: Total number of frames to collect during training

        Returns:
            A SyncDataCollector instance
        """
        collector = SyncDataCollector(
            self.env,
            self.policy,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=self.device,
        )
        return collector

    def create_replay_buffer(self, buffer_size):
        """Create a replay buffer for storing and sampling experiences.

        Args:
            buffer_size: Maximum size of the buffer

        Returns:
            A ReplayBuffer instance
        """
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=buffer_size),
            sampler=SamplerWithoutReplacement(),
        )
        return replay_buffer

    def update(
        self, tensordict_data, num_epochs=NUM_EPOCHS, sub_batch_size=SUB_BATCH_SIZE
    ):
        """Update the policy and value networks using PPO.

        Args:
            tensordict_data: Batch of collected experiences
            num_epochs: Number of optimization epochs per batch
            sub_batch_size: Size of mini-batches for updates

        Returns:
            Dictionary containing training metrics
        """
        metrics = defaultdict(list)

        # Create a replay buffer for the current batch
        replay_buffer = self.create_replay_buffer(tensordict_data.numel())

        # Perform multiple epochs of updates on the same batch of data
        for _ in range(num_epochs):
            # Compute advantage estimates (recompute each epoch as value network changes)
            self.advantage_module(tensordict_data)

            # Reshape data and add to replay buffer
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())

            # Perform mini-batch updates
            batch_size = tensordict_data.numel()
            for _ in range(batch_size // sub_batch_size):
                # Sample a mini-batch without replacement
                subdata = replay_buffer.sample(sub_batch_size)

                # Move data to the correct device
                subdata = subdata.to(self.device)

                # Compute losses
                loss_vals = self.loss_module(subdata)

                # Combine losses
                loss_value = (
                    loss_vals["loss_objective"]  # Policy loss
                    + loss_vals["loss_critic"]  # Value loss
                    + loss_vals["loss_entropy"]  # Entropy bonus
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss_value.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    self.loss_module.parameters(), self.max_grad_norm
                )

                # Perform optimization step
                self.optimizer.step()

                # Record metrics
                for k, v in loss_vals.items():
                    metrics[k].append(v.item())

        # Compute average metrics
        avg_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
        return avg_metrics

    def evaluate(self, n_steps=1000):
        """Evaluate the policy in the environment without exploration.

        Args:
            n_steps: Maximum number of steps for evaluation

        Returns:
            Evaluation results as a tensordict
        """
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_rollout = self.env.rollout(n_steps, self.policy)
        return eval_rollout
