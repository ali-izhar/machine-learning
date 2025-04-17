import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from models import DQN, Transition, ReplayMemory


class SARSAAgent:
    """SARSA (State-Action-Reward-State-Action) agent implementation

    SARSA is an on-policy temporal difference learning algorithm:
    - On-policy: learns Q-values based on actions actually taken under current policy
    - Unlike DQN (off-policy), SARSA evaluates and improves the same policy
    - Update rule: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
    - Key difference: uses Q(s',a') for next action actually taken, not max_a Q(s',a)
    """

    def __init__(
        self,
        state_size,
        action_size,
        device,
        memory,
        batch_size=128,
        gamma=0.99,  # Discount factor
        eps_start=0.9,  # Starting exploration probability
        eps_end=0.05,  # Final exploration probability
        eps_decay=200,  # Decay rate for exploration
        learning_rate=1e-4,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # Initialize Q-network (SARSA uses a single network, unlike DQN's two networks)
        self.q_network = DQN(state_size, action_size).to(device)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.q_network.parameters(), lr=learning_rate, amsgrad=True
        )

        # Initialize steps counter for epsilon calculation
        self.steps_done = 0

        # For storing the last action
        self.last_action = None

    def select_action(self, state, evaluation=False):
        """Select action using epsilon-greedy policy"""
        if evaluation:
            with torch.no_grad():
                return self.q_network(state).max(1)[1].view(1, 1)

        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # Exploit: select best action
                return self.q_network(state).max(1)[1].view(1, 1)
        else:
            # Explore: select random action
            return torch.tensor(
                [[random.randrange(self.action_size)]],
                device=self.device,
                dtype=torch.long,
            )

    def optimize_model(self):
        """Perform one step of optimization using SARSA update rule

        SARSA Bellman equation: Q(s,a) = r + γ * Q(s',a')
        Where a' is the action actually taken in state s' (not the greedy action)
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Create mask for non-final states
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

        # Compute Q(s_t, a_t) - the model computes Q(s_t), then we select actions taken
        state_action_values = self.q_network(state_batch).gather(1, action_batch)

        # Compute next actions for SARSA (on-policy)
        # KEY DIFFERENCE vs DQN: SARSA uses next_actions actually taken from experiences,
        # not the current greedy policy's actions (max Q-value)
        next_actions = torch.zeros(
            self.batch_size, 1, dtype=torch.long, device=self.device
        )
        for i, next_state in enumerate([s for s in batch.next_state if s is not None]):
            # Select action using current policy (epsilon-greedy)
            next_actions[i] = self.select_action(next_state.unsqueeze(0))

        # Compute Q-values for next state-action pairs
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            # SARSA DIFFERENCE: Use Q(s',a') with actual a' instead of max_a Q(s',a)
            # In DQN, we'd use: next_state_values[non_final_mask] = self.q_network(non_final_next_states).max(1)[0]
            next_state_action_values = (
                self.q_network(non_final_next_states)
                .gather(1, next_actions[: non_final_mask.sum()])
                .squeeze()
            )
            next_state_values[non_final_mask] = next_state_action_values

        # Compute expected Q values: γ*Q(s', a') + reward (SARSA Bellman equation)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Calculate loss (Huber loss)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()


def compare_dqn_sarsa(env_name="CartPole-v1", num_episodes=600, render=False):
    """Train both DQN and SARSA agents and compare their performance

    This function illustrates the difference between:
    - DQN: Off-policy, uses max Q-value for updates, more aggressive
    - SARSA: On-policy, uses actual next action, more conservative

    For environments with cliffs/dangers, SARSA tends to learn safer policies,
    while DQN may learn riskier but potentially higher-reward policies.
    """
    import gymnasium as gym
    import matplotlib.pyplot as plt
    from itertools import count
    from .dqn_agent import DQNAgent

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    env = gym.make(env_name)

    # Get environment details
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize memories
    dqn_memory = ReplayMemory(10000)
    sarsa_memory = ReplayMemory(10000)

    # Initialize agents
    dqn_agent = DQNAgent(state_size, action_size, device, dqn_memory)
    sarsa_agent = SARSAAgent(state_size, action_size, device, sarsa_memory)

    # Track episode durations
    dqn_durations = []
    sarsa_durations = []

    # Training loop for DQN
    print("Training DQN agent...")
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = dqn_agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            dqn_memory.push(state, action, next_state, reward)
            state = next_state
            dqn_agent.optimize_model()

            if done:
                dqn_durations.append(t + 1)
                if (i_episode + 1) % 50 == 0:
                    print(f"DQN Episode {i_episode+1}/{num_episodes}, Duration: {t+1}")
                break

    # Training loop for SARSA
    print("Training SARSA agent...")
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # Select first action - SARSA needs the initial action before the first transition
        action = sarsa_agent.select_action(state)

        for t in count():
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
                next_action = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
                # SARSA difference: select the next action now, to include in transition
                next_action = sarsa_agent.select_action(next_state)

            sarsa_memory.push(state, action, next_state, reward)
            sarsa_agent.optimize_model()

            # Update for next step
            state = next_state
            action = next_action

            if done:
                sarsa_durations.append(t + 1)
                if (i_episode + 1) % 50 == 0:
                    print(
                        f"SARSA Episode {i_episode+1}/{num_episodes}, Duration: {t+1}"
                    )
                break

    # Plot results
    plt.figure(figsize=(10, 5))

    # Smooth the curves with moving average
    def moving_average(data, window_size=100):
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    dqn_moving_avg = (
        moving_average(dqn_durations) if len(dqn_durations) > 100 else dqn_durations
    )
    sarsa_moving_avg = (
        moving_average(sarsa_durations)
        if len(sarsa_durations) > 100
        else sarsa_durations
    )

    plt.plot(dqn_moving_avg, label="DQN")
    plt.plot(sarsa_moving_avg, label="SARSA")
    plt.xlabel("Episode")
    plt.ylabel("Duration (moving avg)")
    plt.title("DQN vs SARSA on CartPole-v1")
    plt.legend()
    plt.savefig("dqn_vs_sarsa.png")

    # Print average performance
    print(
        f"DQN Average Duration (last 100 episodes): {np.mean(dqn_durations[-100:]):.2f}"
    )
    print(
        f"SARSA Average Duration (last 100 episodes): {np.mean(sarsa_durations[-100:]):.2f}"
    )

    # Close the environment
    env.close()

    return dqn_durations, sarsa_durations
