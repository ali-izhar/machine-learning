#!/usr/bin/env python
"""DQN implementation for CartPole-v1 environment"""

import os
import sys
import argparse
from train import train, test
from sarsa import compare_dqn_sarsa


def main():
    """Main function to run DQN on CartPole-v1

    DQN algorithm process:
    1. Initialize networks (policy and target)
    2. For each episode:
       a. Reset environment, get initial state
       b. For each step:
          - Select action using epsilon-greedy policy
          - Execute action, observe reward and next state
          - Store transition in replay memory
          - Sample random batch from replay memory
          - Compute target Q-values using target network
          - Update policy network
          - Periodically update target network
    """
    parser = argparse.ArgumentParser(description="Run DQN on CartPole-v1")

    # Mode options
    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument("--train", action="store_true", help="Train the agent")
    mode_group.add_argument("--test", action="store_true", help="Test the agent")
    mode_group.add_argument(
        "--compare", action="store_true", help="Compare DQN with SARSA"
    )

    # Training options
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--episodes", type=int, default=600, help="Number of episodes for training"
    )
    train_group.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    train_group.add_argument(
        "--memory-size", type=int, default=10000, help="Size of replay memory"
    )

    # Testing options
    test_group = parser.add_argument_group("Testing")
    test_group.add_argument(
        "--test-episodes", type=int, default=10, help="Number of episodes for testing"
    )
    test_group.add_argument(
        "--model-path", type=str, default=None, help="Path to load model for testing"
    )

    # General options
    general_group = parser.add_argument_group("General")
    general_group.add_argument(
        "--render", action="store_true", help="Render the environment"
    )
    general_group.add_argument(
        "--save-dir", type=str, default="results", help="Directory to save results"
    )

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # If no flags, do both train and test
    if not args.train and not args.test and not args.compare:
        args.train = True
        args.test = True

    agent = None

    # Compare DQN and SARSA
    if args.compare:
        print("\n" + "=" * 50)
        print("Comparing DQN and SARSA on CartPole-v1...")
        print("=" * 50)
        compare_dqn_sarsa(num_episodes=args.episodes, render=args.render)
        return

    # Train the agent
    if args.train:
        print("\n" + "=" * 50)
        print("Training DQN agent on CartPole-v1...")
        print("=" * 50)
        agent, episode_durations = train(
            num_episodes=args.episodes,
            memory_size=args.memory_size,
            batch_size=args.batch_size,
            render=args.render,
            save_dir=args.save_dir,
        )

    # Test the agent
    if args.test:
        print("\n" + "=" * 50)
        print("Testing DQN agent on CartPole-v1...")
        print("=" * 50)

        model_path = args.model_path
        if model_path is None and agent is None and args.mode == "test":
            # Try to find a saved model in the save directory
            default_path = os.path.join(args.save_dir, "dqn_cartpole.pth")
            if os.path.exists(default_path):
                model_path = default_path
                print(f"Using model found at {model_path}")
            else:
                print("No model path provided and no trained agent available.")
                print(f"Looked for model at {default_path} but it doesn't exist.")
                print("Please train an agent first or provide a model path.")
                sys.exit(1)

        test_durations = test(
            agent=agent,
            model_path=model_path if agent is None else None,
            num_episodes=args.test_episodes,
            render=args.render,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
