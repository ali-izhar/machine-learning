"""
TD(0) Implementation
===================

This module implements TD(0) learning algorithm for policy evaluation.

TD(0) Algorithm Pseudocode:
--------------------------
Input: 
    - policy π to be evaluated
    - Initial state-value function V (can be arbitrary except terminal states = 0)

Algorithm Parameters:
    - α: step-size parameter (learning rate)
    - γ: discount factor
    - num_episodes: number of episodes to run

Initialize:
    V(s) arbitrarily for all s ∈ S
    V(terminal) = 0

For each episode:
    Initialize S
    For each step of episode:
        A ← action given by π for S
        Take action A, observe R, S'
        V(S) ← V(S) + α[R + γV(S') - V(S)]
        S ← S'
    Until S is terminal

The key difference between TD(0) and Monte Carlo:
- TD(0) updates V(S) at each step using bootstrapping
- MC waits until episode end to get actual return G
"""

from typing import Callable, Dict, Any


class TD0:
    def __init__(
        self, alpha: float = 0.1, gamma: float = 1.0, num_episodes: int = 1000
    ):
        """
        Initialize TD(0) learner.

        Args:
            alpha: Learning rate (step-size parameter)
            gamma: Discount factor
            num_episodes: Number of episodes to run
        """
        self.alpha = alpha
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.V = {}  # State-value function

    def learn(self, env: Any, policy: Callable, initial_value: float = 0.0) -> Dict:
        """
        Run TD(0) algorithm to evaluate a policy.

        Args:
            env: Environment that implements:
                - reset(): Reset environment and return initial state
                - step(action): Take action and return (next_state, reward, done, info)
            policy: Function that takes state and returns action
            initial_value: Initial value for V(s)

        Returns:
            Learned state-value function V
        """
        # Initialize V(s)
        self.V = {}

        for episode in range(self.num_episodes):
            # Initialize S
            state = env.reset()
            done = False

            while not done:
                # Get action from policy
                action = policy(state)

                # Take action, observe R, S'
                next_state, reward, done, _ = env.step(action)

                # TD(0) update
                # V(S) ← V(S) + α[R + γV(S') - V(S)]
                if state not in self.V:
                    self.V[state] = initial_value
                if next_state not in self.V:
                    self.V[next_state] = initial_value

                # If terminal state, V(S') = 0
                next_value = 0 if done else self.V[next_state]

                # TD(0) update rule
                self.V[state] = self.V[state] + self.alpha * (
                    reward + self.gamma * next_value - self.V[state]
                )

                # Move to next state
                state = next_state

            # Print progress
            if (episode + 1) % 100 == 0:
                print(f"Completed {episode + 1} episodes")

        return self.V


def example_usage():
    """
    Example showing how to use TD(0) with a simple environment.
    """

    # Example policy (replace with actual policy)
    def simple_policy(state):
        return 0  # Always take action 0

    # Create TD(0) learner
    td0 = TD0(alpha=0.1, gamma=1.0, num_episodes=1000)

    # Run TD(0) learning
    # V = td0.learn(env, simple_policy)

    # Print learned value function
    # print("Learned Value Function:")
    # for state, value in V.items():
    #     print(f"State {state}: {value:.3f}")


if __name__ == "__main__":
    example_usage()
