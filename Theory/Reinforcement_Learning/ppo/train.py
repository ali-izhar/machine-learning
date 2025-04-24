"""Train PPO agent on Inverted Double Pendulum environment"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type

from ppo_agent import PPOAgent

# Default parameters
FRAMES_PER_BATCH = 1000
TOTAL_FRAMES = 500000  # Reduced from 1M for faster training
NUM_CELLS = 256
LEARNING_RATE = 3e-4
CLIP_EPSILON = 0.2
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_EPS = 1e-4
MAX_GRAD_NORM = 1.0
NUM_EPOCHS = 10
SUB_BATCH_SIZE = 64
EVAL_FREQ = 10
NUM_EVAL_EPISODES = 5


def train(
    env_name="InvertedDoublePendulum-v4",
    frames_per_batch=FRAMES_PER_BATCH,
    total_frames=TOTAL_FRAMES,
    num_cells=NUM_CELLS,
    lr=LEARNING_RATE,
    clip_epsilon=CLIP_EPSILON,
    gamma=GAMMA,
    lmbda=LAMBDA,
    entropy_eps=ENTROPY_EPS,
    max_grad_norm=MAX_GRAD_NORM,
    num_epochs=NUM_EPOCHS,
    sub_batch_size=SUB_BATCH_SIZE,
    eval_freq=EVAL_FREQ,
    save_dir="results",
    seed=None,
):
    """
    Train a PPO agent on a Gymnasium environment.

    Args:
        env_name: Name of the Gymnasium environment to train on
        frames_per_batch: Number of environment steps to collect per batch
        total_frames: Total number of environment steps for training
        num_cells: Number of cells in hidden layers
        lr: Learning rate
        clip_epsilon: PPO clipping parameter
        gamma: Discount factor
        lmbda: GAE lambda parameter
        entropy_eps: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm for clipping
        num_epochs: Number of optimization epochs per batch
        sub_batch_size: Size of mini-batches for updates
        eval_freq: Frequency of evaluation (in batch updates)
        save_dir: Directory to save results
        seed: Random seed

    Returns:
        agent: Trained PPO agent
        logs: Dictionary with training metrics
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create and prepare environment
    print(f"Creating environment: {env_name}")
    base_env = GymEnv(env_name, device=device)

    # Apply transforms
    env = TransformedEnv(
        base_env,
        Compose(
            # Normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )

    # Initialize observation normalization
    print("Initializing observation normalization...")
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    # Verify environment specs
    check_env_specs(env)

    # Print environment specs
    print(f"Observation shape: {env.observation_spec['observation'].shape}")
    print(f"Action shape: {env.action_spec.shape}")

    # Create PPO agent
    print("Creating PPO agent...")
    agent = PPOAgent(
        env=env,
        device=device,
        num_cells=num_cells,
        clip_epsilon=clip_epsilon,
        gamma=gamma,
        lmbda=lmbda,
        entropy_eps=entropy_eps,
        lr=lr,
        max_grad_norm=max_grad_norm,
    )

    # Create data collector
    collector = agent.create_collector(frames_per_batch, total_frames)

    # Training loop
    print("Starting training...")
    start_time = time.time()

    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""

    # Collect data and perform updates
    for i, tensordict_data in enumerate(collector):
        # Update the policy with collected data
        train_metrics = agent.update(
            tensordict_data, num_epochs=num_epochs, sub_batch_size=sub_batch_size
        )

        # Record training metrics
        for k, v in train_metrics.items():
            logs[k].append(v)

        # Record rewards
        batch_reward = tensordict_data["next", "reward"].mean().item()
        logs["batch_reward"].append(batch_reward)

        # Record step counts
        max_step_count = tensordict_data["step_count"].max().item()
        logs["step_count"].append(max_step_count)

        # Update progress bar
        pbar.update(tensordict_data.numel())
        reward_str = f"reward={batch_reward:.2f}"
        step_str = f"steps={max_step_count}"

        # Evaluate the policy periodically
        if i % eval_freq == 0:
            print(f"\nEvaluation at batch {i}...")
            eval_rewards = []
            eval_steps = []

            for _ in range(NUM_EVAL_EPISODES):
                eval_rollout = agent.evaluate()
                eval_reward = eval_rollout["next", "reward"].sum().item()
                eval_steps.append(eval_rollout["step_count"].max().item())
                eval_rewards.append(eval_reward)

            # Record evaluation metrics
            avg_eval_reward = np.mean(eval_rewards)
            avg_eval_steps = np.mean(eval_steps)
            logs["eval_reward"].append(avg_eval_reward)
            logs["eval_steps"].append(avg_eval_steps)

            eval_str = (
                f"eval_reward={avg_eval_reward:.2f}, eval_steps={avg_eval_steps:.1f}"
            )

            # Save model checkpoint
            torch.save(
                {
                    "policy_state_dict": agent.policy.state_dict(),
                    "value_state_dict": agent.value.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                },
                f"{save_dir}/checkpoint_{i}.pt",
            )

        # Update progress bar description
        pbar.set_description(f"Batch {i}: {reward_str}, {step_str}, {eval_str}")

    # Calculate training time
    total_time = time.time() - start_time
    print(f"Training complete! Time: {total_time:.2f}s")

    # Plot training curves
    plot_training_curves(logs, save_dir)

    # Save final model
    torch.save(
        {
            "policy_state_dict": agent.policy.state_dict(),
            "value_state_dict": agent.value.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
        },
        f"{save_dir}/model_final.pt",
    )

    return agent, logs


def test(
    env_name="InvertedDoublePendulum-v4",
    checkpoint_path=None,
    num_episodes=10,
    render=True,
    save_dir="results",
    seed=None,
):
    """
    Test a trained PPO agent.

    Args:
        env_name: Name of the Gymnasium environment
        checkpoint_path: Path to the saved model checkpoint
        num_episodes: Number of episodes to test
        render: Whether to render the environment
        save_dir: Directory to save results
        seed: Random seed

    Returns:
        Dictionary with test metrics
    """
    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    render_mode = "human" if render else None
    base_env = GymEnv(env_name, device=device, render_mode=render_mode)

    # Apply transforms
    env = TransformedEnv(
        base_env,
        Compose(
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )

    # Initialize observation normalization
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    # Create agent
    agent = PPOAgent(env=env, device=device)

    # Load saved model if provided
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        agent.value.load_state_dict(checkpoint["value_state_dict"])
        print(f"Loaded model from {checkpoint_path}")
    else:
        # Try to find the latest checkpoint in save_dir
        checkpoint_files = [
            f for f in os.listdir(save_dir) if f.startswith("checkpoint_")
        ]
        if checkpoint_files:
            latest_checkpoint = max(
                checkpoint_files, key=lambda x: int(x.split("_")[1].split(".")[0])
            )
            checkpoint_path = os.path.join(save_dir, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            agent.policy.load_state_dict(checkpoint["policy_state_dict"])
            agent.value.load_state_dict(checkpoint["value_state_dict"])
            print(f"Loaded model from {checkpoint_path}")
        else:
            print("No checkpoint found, using untrained model")

    # Test loop
    rewards = []
    steps = []

    for episode in range(num_episodes):
        # Run evaluation episode
        eval_rollout = agent.evaluate()

        # Record metrics
        episode_reward = eval_rollout["next", "reward"].sum().item()
        episode_steps = eval_rollout["step_count"].max().item()

        rewards.append(episode_reward)
        steps.append(episode_steps)

        print(
            f"Episode {episode+1}/{num_episodes}: Reward = {episode_reward:.2f}, Steps = {episode_steps}"
        )

    # Compute statistics
    mean_reward = np.mean(rewards)
    mean_steps = np.mean(steps)

    print(f"\nTest Results:")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Mean steps: {mean_steps:.2f}")

    # Create simple bar plot of rewards
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, num_episodes + 1), rewards)
    plt.axhline(
        y=mean_reward, color="r", linestyle="-", label=f"Mean: {mean_reward:.2f}"
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Test Results")
    plt.legend()
    plt.savefig(f"{save_dir}/test_results.png")

    test_results = {
        "rewards": rewards,
        "steps": steps,
        "mean_reward": mean_reward,
        "mean_steps": mean_steps,
    }

    return test_results


def plot_training_curves(logs, save_dir):
    """
    Plot training curves and save them to disk.

    Args:
        logs: Dictionary containing training metrics
        save_dir: Directory to save plots
    """
    plt.figure(figsize=(15, 10))

    # Plot training rewards
    plt.subplot(2, 2, 1)
    plt.plot(logs["batch_reward"])
    plt.title("Training Batch Rewards")
    plt.xlabel("Batch")
    plt.ylabel("Average Reward")

    # Plot evaluation rewards
    if "eval_reward" in logs and logs["eval_reward"]:
        plt.subplot(2, 2, 2)
        plt.plot(
            np.arange(0, len(logs["batch_reward"]), EVAL_FREQ)[
                : len(logs["eval_reward"])
            ],
            logs["eval_reward"],
        )
        plt.title("Evaluation Rewards")
        plt.xlabel("Batch")
        plt.ylabel("Total Reward")

    # Plot step counts
    plt.subplot(2, 2, 3)
    plt.plot(logs["step_count"])
    plt.title("Training Step Counts")
    plt.xlabel("Batch")
    plt.ylabel("Max Steps")

    # Plot evaluation steps
    if "eval_steps" in logs and logs["eval_steps"]:
        plt.subplot(2, 2, 4)
        plt.plot(
            np.arange(0, len(logs["batch_reward"]), EVAL_FREQ)[
                : len(logs["eval_steps"])
            ],
            logs["eval_steps"],
        )
        plt.title("Evaluation Step Counts")
        plt.xlabel("Batch")
        plt.ylabel("Average Steps")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/learning_curves.png")
    print(f"Saved training curves to {save_dir}/learning_curves.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test PPO agent")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Mode: train or test",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="InvertedDoublePendulum-v4",
        help="Gymnasium environment name",
    )
    parser.add_argument(
        "--frames", type=int, default=TOTAL_FRAMES, help="Total frames for training"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file for testing",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes for testing"
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Disable rendering during testing"
    )
    parser.add_argument(
        "--save-dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    if args.mode == "train":
        train(
            env_name=args.env,
            total_frames=args.frames,
            save_dir=args.save_dir,
            seed=args.seed,
        )
    else:
        test(
            env_name=args.env,
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes,
            render=not args.no_render,
            save_dir=args.save_dir,
            seed=args.seed,
        )
