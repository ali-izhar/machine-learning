import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from itertools import count
from models import ReplayMemory, DQN
from dqn_agent import DQNAgent
from utils import plot_durations


def train(
    num_episodes=600,
    memory_size=10000,
    batch_size=128,
    render=False,
    save_dir="results",
):
    """Train DQN agent on CartPole-v1

    This function implements the core DQN training loop:
    1. For each episode, reset environment
    2. For each step in episode:
        - Select action (epsilon-greedy)
        - Execute action in environment
        - Observe reward and next state
        - Store experience
        - Update neural network (if enough samples)
    3. Track progress and visualize learning

    Args:
        num_episodes: Number of episodes to train
        memory_size: Size of replay memory
        batch_size: Batch size for training
        render: Whether to render the environment
        save_dir: Directory to save results
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    env = gym.make("CartPole-v1", render_mode="rgb_array" if render else None)

    # Get environment details
    state_size = env.observation_space.shape[0]  # 4 for CartPole
    action_size = env.action_space.n  # 2 for CartPole

    # Create replay memory
    memory = ReplayMemory(memory_size)

    # Create DQN agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        device=device,
        memory=memory,
        batch_size=batch_size,
    )

    # Track episode durations
    episode_durations = []

    # Training start time
    start_time = time.time()

    # Main training loop
    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        # Convert state to tensor and add batch dimension (batch_size=1)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # Episode loop
        for t in count():
            # Select and perform an action
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            # Process next state
            if terminated:
                # For terminal states, there is no next state
                # This helps the agent learn that terminal states have no future reward
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Store the transition in memory
            # This experience will be randomly sampled during training
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of optimization
            # This trains the neural network on a random batch from memory
            agent.optimize_model()

            # End episode if done
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)

                # Print progress every 10 episodes
                if (i_episode + 1) % 10 == 0:
                    avg_duration = (
                        np.mean(episode_durations[-10:])
                        if len(episode_durations) >= 10
                        else np.mean(episode_durations)
                    )
                    print(
                        f"Episode {i_episode+1}/{num_episodes}, Duration: {t+1}, Avg Last 10: {avg_duration:.2f}"
                    )
                break

    # Training end time
    end_time = time.time()
    print(f"Training complete! Time taken: {end_time - start_time:.2f} seconds")

    # Save the final plot
    plt.figure(figsize=(10, 5))
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title("Training Results")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())

    # Plot 100-episode moving average
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
        plt.legend(["Episode Duration", "100-episode Moving Average"])
    else:
        plt.legend(["Episode Duration"])

    plt.savefig(f"{save_dir}/training_results.png")
    print(f"Results saved to {save_dir}/training_results.png")

    # Save model
    torch.save(agent.policy_net.state_dict(), f"{save_dir}/dqn_cartpole.pth")
    print(f"Model saved to {save_dir}/dqn_cartpole.pth")

    # Close the environment
    env.close()

    return agent, episode_durations


def test(agent=None, model_path=None, num_episodes=10, render=True, save_dir="results"):
    """Test a trained DQN agent or a model from a saved path

    During testing:
    - No exploration is used (always choose best action)
    - No learning updates are performed
    - The episode length and cumulative reward are recorded
    - The environment can be rendered for visualization

    Args:
        agent: Trained DQN agent (if None, will load from model_path)
        model_path: Path to saved model (used if agent is None)
        num_episodes: Number of episodes to test
        render: Whether to render the environment
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    # Get environment details
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create or load agent
    if agent is None:
        if model_path is None:
            raise ValueError("Either agent or model_path must be provided")

        # Create new agent
        policy_net = DQN(state_size, action_size).to(device)
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        policy_net.eval()  # Set to evaluation mode (no gradient updates)

        # Create temporary agent just for inference
        class TempAgent:
            def __init__(self, policy_net):
                self.policy_net = policy_net

            def select_action(self, state):
                # During testing, we always choose the best action (no exploration)
                with torch.no_grad():
                    return self.policy_net(state).max(1)[1].view(1, 1)

        agent = TempAgent(policy_net)

    # Track episode durations
    test_durations = []

    print("Testing agent...")
    for i_episode in range(num_episodes):
        # Initialize the environment
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # Episode loop
        episode_reward = 0
        for t in count():
            # Select action (no exploration)
            action = agent.select_action(state)

            # Take action
            observation, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward
            done = terminated or truncated

            if done:
                test_durations.append(t + 1)
                print(f"Episode {i_episode+1}/{num_episodes}, Duration: {t+1}")
                break

            # Update state
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            state = next_state

    # Print test results
    avg_duration = np.mean(test_durations)
    print(f"\nTest Results:")
    print(f"Average Duration: {avg_duration:.2f}")
    print(f"Episode Durations: {test_durations}")

    # Save test results
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, num_episodes + 1), test_durations)
    plt.axhline(
        y=avg_duration, color="r", linestyle="-", label=f"Average: {avg_duration:.2f}"
    )
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.title("Test Results")
    plt.legend()
    plt.savefig(f"{save_dir}/test_results.png")
    print(f"Test results saved to {save_dir}/test_results.png")

    # Close the environment
    env.close()

    return test_durations
