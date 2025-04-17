"""Train DDPG agent on Pendulum environment"""

import gymnasium as gym
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import time
import os
from collections import deque

from ddpg_agent import Agent


def train(
    env_name="Pendulum-v1",
    n_episodes=1000,
    max_t=300,
    print_every=100,
    seed=2,
    save_dir="results",
):
    """Train DDPG agent on Pendulum environment

    This function implements the core DDPG training loop:
    1. For each episode, reset environment and agent
    2. For each step in episode:
        - Select action based on current policy with noise for exploration
        - Execute action in environment
        - Observe reward and next state
        - Store experience and update networks
    3. Track progress and visualize learning

    Args:
        env_name: Name of the environment to train on
        n_episodes: Number of episodes to train
        max_t: Maximum number of timesteps per episode
        print_every: How often to print progress
        seed: Random seed for reproducibility
        save_dir: Directory to save results
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create environment
    env = gym.make(env_name)

    # Set random seeds for reproducibility
    env.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get environment details
    state_size = env.observation_space.shape[0]  # 3 for Pendulum
    action_size = env.action_space.shape[0]  # 1 for Pendulum

    # Create DDPG agent
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed)

    # For tracking progress
    scores_deque = deque(maxlen=print_every)
    scores = []

    # Training start time
    start_time = time.time()

    print(f"Training DDPG agent on {env_name}...")
    print(f"State size: {state_size}, Action size: {action_size}")

    # Main training loop
    for i_episode in range(1, n_episodes + 1):
        # Reset environment and get initial state
        state, _ = env.reset()
        # Reset the noise process for exploration
        agent.reset()

        # Episode cumulative reward
        score = 0

        # Episode loop
        for t in range(max_t):
            # Select an action (with noise for exploration)
            action = agent.act(state)

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store experience and learn
            agent.step(state, action, reward, next_state, done)

            # Update state and score
            state = next_state
            score += reward

            # End episode if done
            if done:
                break

        # Save score and track progress
        scores_deque.append(score)
        scores.append(score)

        # Print progress
        print(
            f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}", end=""
        )

        # Save checkpoint every episode
        torch.save(agent.actor_local.state_dict(), f"{save_dir}/checkpoint_actor.pth")
        torch.save(agent.critic_local.state_dict(), f"{save_dir}/checkpoint_critic.pth")

        # Print detailed progress every print_every episodes
        if i_episode % print_every == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}")

    # Training end time
    end_time = time.time()
    print(f"Training complete! Time taken: {end_time - start_time:.2f} seconds")

    # Plot the scores
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.title(f"DDPG Training Results - {env_name}")
    plt.savefig(f"{save_dir}/training_results.png")
    print(f"Training plot saved to {save_dir}/training_results.png")

    # Close the environment
    env.close()

    return agent, scores


def test(
    agent=None,
    actor_path=None,
    critic_path=None,
    env_name="Pendulum-v1",
    n_episodes=3,
    max_t=300,
    seed=2,
    render=True,
    save_dir="results",
):
    """Test a trained DDPG agent

    During testing:
    - No exploration noise is used (deterministic policy)
    - No learning updates are performed
    - The episode length and cumulative reward are recorded
    - The environment can be rendered for visualization

    Args:
        agent: Trained DDPG agent (if None, will load from model_paths)
        actor_path: Path to saved actor model
        critic_path: Path to saved critic model
        env_name: Name of the environment to test on
        n_episodes: Number of episodes to test
        max_t: Maximum number of timesteps per episode
        seed: Random seed for reproducibility
        render: Whether to render the environment
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create environment
    env = gym.make(env_name, render_mode="human" if render else None)

    # Set random seeds for reproducibility
    env.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get environment details
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # Create or load agent
    if agent is None:
        if actor_path is None or critic_path is None:
            # Default to the saved checkpoints if paths not provided
            actor_path = f"{save_dir}/checkpoint_actor.pth"
            critic_path = f"{save_dir}/checkpoint_critic.pth"

        agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed)
        agent.actor_local.load_state_dict(torch.load(actor_path))
        agent.critic_local.load_state_dict(torch.load(critic_path))

    # Track episode scores
    test_scores = []

    print(f"Testing DDPG agent on {env_name}...")

    # Test loop
    for i_episode in range(1, n_episodes + 1):
        # Reset environment
        state, _ = env.reset()

        # Episode cumulative reward
        score = 0

        # Episode loop
        for t in range(max_t):
            # Select action (without noise for deterministic policy)
            action = agent.act(state, add_noise=False)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update state and score
            state = next_state
            score += reward

            # End episode if done
            if done:
                break

        # Save score
        test_scores.append(score)
        print(f"Episode {i_episode}/{n_episodes}, Score: {score:.2f}")

    # Print test results
    avg_score = np.mean(test_scores)
    print(f"\nTest Results:")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Episode Scores: {test_scores}")

    # Close the environment
    env.close()

    return test_scores


if __name__ == "__main__":
    # Example usage:
    # Train a DDPG agent
    agent, scores = train()

    # Test the trained agent
    test(agent=agent)
