"""
Blackjack Monte Carlo Policy Evaluation
=======================================

This implementation demonstrates Monte Carlo methods for policy evaluation in the blackjack environment.

Mathematical Foundation
----------------------
Monte Carlo methods estimate value functions by averaging returns from complete episodes:

For a state s, the value function V(s) is defined as the expected return:
    V(s) = E[G_t | S_t = s]
where G_t is the total discounted return from time step t.

For an episode with states (s_0, s_1, ..., s_T) and rewards (r_1, r_2, ..., r_T),
the return following state s_t is:
    G_t = r_{t+1} + r_{t+2} + ... + r_T  (γ=1, no discounting in this implementation)

State Values vs. Action Values (V vs. Q)
----------------------------------------
This implementation includes both state-value (V) and action-value (Q) estimation:

State-value function V(s):
- Represents the expected return starting from state s and following policy π
- V(s) = E_π[G_t | S_t = s]

Action-value function Q(s,a):
- Represents the expected return starting from state s, taking action a, and following policy π thereafter
- Q(s,a) = E_π[G_t | S_t = s, A_t = a]

Advantages of Q-values:
- Directly inform policy improvement without requiring a model of the environment
- Enable model-free control by determining the best action in each state
- Essential for control algorithms like Q-learning and SARSA

First-Visit Monte Carlo
----------------------
The first-visit estimator only counts the return following the first occurrence of state s in each episode:
    V(s) = (1/N(s)) * ∑ G_t[i]
where:
- N(s) is the number of episodes in which state s was visited at least once
- G_t[i] is the return following the first occurrence of s in the i-th episode

Algorithm:
1. Initialize: 
   - V(s) = 0 for all s ∈ S
   - Returns(s) = empty list for all s ∈ S
2. For each episode:
   - Generate episode following policy π: S_0, A_0, R_1, S_1, ..., S_T
   - G ← 0
   - For t = T-1, T-2, ..., 0:
      - G ← G + R_{t+1}
      - If S_t not in S_0, S_1, ..., S_{t-1} (first visit to S_t):
         - N(S_t) ← N(S_t) + 1
         - Returns(S_t) ← Returns(S_t) + G
         - V(S_t) ← Returns(S_t) / N(S_t)

For Q-values, the same approach is used but for state-action pairs:
    Q(s,a) = (1/N(s,a)) * ∑ G_t[i]
where:
- N(s,a) is the number of episodes in which the state-action pair (s,a) was visited at least once
- G_t[i] is the return following the first occurrence of (s,a) in the i-th episode

Every-Visit Monte Carlo
-----------------------
The every-visit estimator counts every occurrence of state s in each episode:
    V(s) = (1/N'(s)) * ∑ G_t[i,j]
where:
- N'(s) is the total number of times state s was visited across all episodes
- G_t[i,j] is the return following the j-th occurrence of s in the i-th episode

Algorithm:
1. Initialize: 
   - V(s) = 0 for all s ∈ S
   - Returns(s) = empty list for all s ∈ S
2. For each episode:
   - Generate episode following policy π: S_0, A_0, R_1, S_1, ..., S_T
   - G ← 0
   - For t = T-1, T-2, ..., 0:
      - G ← G + R_{t+1}
      - N(S_t) ← N(S_t) + 1
      - Returns(S_t) ← Returns(S_t) + G
      - V(S_t) ← Returns(S_t) / N(S_t)

Convergence Properties
---------------------
Both first-visit and every-visit MC converge to the true value function V_π(s) as the number of visits to each state approaches infinity.

For first-visit MC, this is guaranteed by the law of large numbers, as each return is an independent, identically distributed estimate of V_π(s).

Every-visit MC is biased in finite samples because returns following multiple visits to s within an episode are correlated. However, this bias disappears asymptotically.

Blackjack-Specific Considerations
--------------------------------
In the blackjack problem:
- States are represented as (player_sum, dealer_showing, usable_ace)
- Terminal rewards are +1 (win), 0 (draw), -1 (lose)
- A simple policy is used: stick on 20-21, hit otherwise
- No discounting is applied (γ=1)
- Episodes naturally terminate (no need for arbitrary truncation)

Mathematically, the blackjack problem has a reward of -1 for going bust, and 
a terminal reward of +1, 0, or -1 depending on the final outcome.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from collections import namedtuple

# Define a namedtuple for storing episode statistics
EpisodeStats = namedtuple("EpisodeStats", ["episode_lengths", "episode_rewards"])


class BlackjackEnv:
    """
    Blackjack environment following the problem formulation.

    States:
    - Player sum: 12-21 (for sum < 12, always hit)
    - Dealer showing card: Ace-10 (Ace=1, J/Q/K=10)
    - Usable ace: Yes/No (whether player has a usable ace)

    Actions:
    - 0: Stick
    - 1: Hit

    Rewards:
    - Win: +1
    - Lose: -1
    - Draw: 0
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the environment for a new episode."""
        # Player's state
        self.player_sum = 0
        self.player_has_ace = False
        self.player_has_natural = False

        # Dealer's state
        self.dealer_sum = 0
        self.dealer_has_ace = False
        self.dealer_has_natural = False
        self.dealer_showing = 0

        # Deal initial cards
        self._deal_initial()

        return self._get_state()

    def _get_state(self):
        """Return the current state tuple."""
        return (self.player_sum, self.dealer_showing, int(self.player_has_ace))

    def _deal_card(self):
        """Draw a card from the deck (with replacement)."""
        # Card values: A=1, 2-10=face value, J/Q/K=10
        card = min(np.random.randint(1, 14), 10)
        return card

    def _deal_initial(self):
        """Deal initial cards to player and dealer."""
        # Deal player's cards
        card1 = self._deal_card()
        card2 = self._deal_card()

        # Handle aces
        self.player_has_ace = card1 == 1 or card2 == 1

        # Calculate sum (convert ace to 11 if possible)
        self.player_sum = card1 + card2
        if self.player_has_ace and self.player_sum + 10 <= 21:
            self.player_sum += 10

        # Check for natural
        self.player_has_natural = self.player_sum == 21

        # Deal dealer's cards
        dealer_card1 = self._deal_card()
        dealer_card2 = self._deal_card()

        # Handle dealer's aces
        self.dealer_has_ace = dealer_card1 == 1 or dealer_card2 == 1

        # Calculate dealer's sum
        self.dealer_sum = dealer_card1 + dealer_card2
        if self.dealer_has_ace and self.dealer_sum + 10 <= 21:
            self.dealer_sum += 10

        # Check for dealer natural
        self.dealer_has_natural = self.dealer_sum == 21

        # Set dealer's showing card (first card)
        self.dealer_showing = dealer_card1

    def step(self, action):
        """
        Take an action in the environment.

        Args:
            action: 0 (stick) or 1 (hit)

        Returns:
            next_state: The next state
            reward: The reward received
            done: Whether the episode is done
            info: Additional info
        """
        done = False
        reward = 0

        if action == 1:  # Hit
            card = self._deal_card()

            # Update player's sum
            self.player_sum += card

            # Check if player got an ace
            if card == 1 and self.player_sum + 10 <= 21:
                self.player_sum += 10
                self.player_has_ace = True

            # Check for bust
            if self.player_sum > 21:
                # Check if can convert ace from 11 to 1
                if self.player_has_ace and self.player_sum > 21:
                    self.player_sum -= 10
                    self.player_has_ace = False

                # If still bust after converting ace
                if self.player_sum > 21:
                    reward = -1
                    done = True

        else:  # Stick
            done = True

            # Now it's dealer's turn
            while self.dealer_sum < 17:
                card = self._deal_card()
                self.dealer_sum += card

                # Handle ace
                if card == 1 and self.dealer_sum + 10 <= 21:
                    self.dealer_sum += 10
                    self.dealer_has_ace = True

                # Check if need to convert dealer's ace from 11 to 1
                if self.dealer_has_ace and self.dealer_sum > 21:
                    self.dealer_sum -= 10
                    self.dealer_has_ace = False

            # Determine outcome
            if self.dealer_sum > 21:  # Dealer bust
                reward = 1
            else:
                if self.player_sum > self.dealer_sum:
                    reward = 1
                elif self.player_sum < self.dealer_sum:
                    reward = -1
                else:  # Draw
                    reward = 0

        return self._get_state(), reward, done, {}

    def generate_episode(self, policy):
        """
        Generate an episode using policy.

        Args:
            policy: Function that takes state and returns action

        Returns:
            List of (state, action, reward) tuples
        """
        episode = []
        state = self.reset()

        # Handle cases where player sum < 12 (always hit)
        while state[0] < 12:
            next_state, reward, done, _ = self.step(1)  # Hit
            if done:
                episode.append((state, 1, reward))
                return episode
            state = next_state

        # Continue with policy for sum >= 12
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, _ = self.step(action)
            episode.append((state, action, reward))
            state = next_state

        return episode


def simple_policy(state):
    """Simple policy: stick on 20 or 21, otherwise hit."""
    player_sum = state[0]
    return 0 if player_sum >= 20 else 1


def policy_stick_above_18(state):
    """Policy: stick if the player's sum is > 18, otherwise hit."""
    player_sum = state[0]
    return 0 if player_sum > 18 else 1


def policy_stick_above_17_prob_hit(state):
    """Policy: stick if the player's sum is > 17, otherwise hit with probability 30%."""
    player_sum = state[0]
    if player_sum > 17:
        return 0  # Stick
    else:
        # Hit with 30% probability, stick with 70% probability
        return 1 if np.random.random() < 0.3 else 0


def create_epsilon_greedy_policy(Q, epsilon, nA=2):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
           Each value is a numpy array of length nA (see nA below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(state):
        # Initialize action probabilities with epsilon/nA for all actions
        A = np.ones(nA) * epsilon / nA

        # Get Q-values for this state
        q_values = np.array([Q.get((state, a), 0.0) for a in range(nA)])

        # If all Q-values are equal, use a simple heuristic for blackjack
        if np.all(q_values == q_values[0]):
            best_action = (
                1 if state[0] < 17 else 0
            )  # Hit if player sum < 17, else stick
        else:
            best_action = np.argmax(q_values)

        # Add extra probability to the best action
        A[best_action] += 1.0 - epsilon

        return A

    return policy_fn


def monte_carlo_first_visit(env, policy, num_episodes=10000):
    """
    First-visit Monte Carlo policy evaluation.

    Args:
        env: The environment
        policy: Function that takes state and returns action
        num_episodes: Number of episodes to simulate

    Returns:
        Estimated state-value function V(s) and episode statistics
    """
    # Initialize returns and counts
    returns_sum = {}  # sum of returns for each state
    returns_count = {}  # count of returns for each state
    V = {}  # value function

    # Initialize statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes)
    )

    for episode_idx in range(num_episodes):
        # Generate an episode
        episode = env.generate_episode(policy)

        # Track episode statistics
        stats.episode_lengths[episode_idx] = len(episode)
        stats.episode_rewards[episode_idx] = sum(reward for _, _, reward in episode)

        # Get returns and update value function
        G = 0
        # Process states in reverse order
        visited_states = set()

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + G  # Calculate return (no discount in this problem)

            # If this is the first visit to the state
            if state not in visited_states:
                visited_states.add(state)

                # Update returns
                if state not in returns_sum:
                    returns_sum[state] = 0
                    returns_count[state] = 0

                returns_sum[state] += G
                returns_count[state] += 1

                # Update value function
                V[state] = returns_sum[state] / returns_count[state]

        # Print progress
        if (episode_idx + 1) % 1000 == 0:
            print(f"Completed {episode_idx + 1} episodes (first-visit MC)")

    return V, stats


def monte_carlo_every_visit(env, policy, num_episodes=10000):
    """
    Every-visit Monte Carlo policy evaluation.

    Args:
        env: The environment
        policy: Function that takes state and returns action
        num_episodes: Number of episodes to simulate

    Returns:
        Estimated state-value function V(s) and episode statistics
    """
    # Initialize returns and counts
    returns_sum = {}  # sum of returns for each state
    returns_count = {}  # count of returns for each state
    V = {}  # value function

    # Initialize statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes)
    )

    for episode_idx in range(num_episodes):
        # Generate an episode
        episode = env.generate_episode(policy)

        # Track episode statistics
        stats.episode_lengths[episode_idx] = len(episode)
        stats.episode_rewards[episode_idx] = sum(reward for _, _, reward in episode)

        # Get returns and update value function
        G = 0
        # Process states in reverse order
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + G  # Calculate return (no discount in this problem)

            # Update returns for every visit
            if state not in returns_sum:
                returns_sum[state] = 0
                returns_count[state] = 0

            returns_sum[state] += G
            returns_count[state] += 1

            # Update value function
            V[state] = returns_sum[state] / returns_count[state]

        # Print progress
        if (episode_idx + 1) % 1000 == 0:
            print(f"Completed {episode_idx + 1} episodes (every-visit MC)")

    return V, stats


def monte_carlo_q_evaluation(env, policy, num_episodes=10000):
    """
    First-visit Monte Carlo action-value (Q) evaluation.

    Args:
        env: The environment
        policy: Function that takes state and returns action
        num_episodes: Number of episodes to simulate

    Returns:
        Estimated action-value function Q(s,a) and episode statistics
    """
    # Initialize returns and counts
    returns_sum = {}  # sum of returns for each state-action pair
    returns_count = {}  # count of returns for each state-action pair
    Q = {}  # action-value function

    # Initialize statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes)
    )

    for episode_idx in range(num_episodes):
        # Generate an episode
        episode = env.generate_episode(policy)

        # Track episode statistics
        stats.episode_lengths[episode_idx] = len(episode)
        stats.episode_rewards[episode_idx] = sum(reward for _, _, reward in episode)

        # Get returns and update value function
        G = 0
        # Process states in reverse order
        visited_pairs = set()

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + G  # Calculate return (no discount in this problem)

            # Create state-action pair
            sa_pair = (state, action)

            # If this is the first visit to the state-action pair
            if sa_pair not in visited_pairs:
                visited_pairs.add(sa_pair)

                # Update returns
                if sa_pair not in returns_sum:
                    returns_sum[sa_pair] = 0
                    returns_count[sa_pair] = 0

                returns_sum[sa_pair] += G
                returns_count[sa_pair] += 1

                # Update Q-value function
                Q[sa_pair] = returns_sum[sa_pair] / returns_count[sa_pair]

        # Print progress
        if (episode_idx + 1) % 1000 == 0:
            print(f"Completed {episode_idx + 1} episodes (Q-value MC)")

    return Q, stats


def monte_carlo_control_epsilon_greedy(
    env, num_episodes=10000, epsilon=0.1, discount_factor=1.0
):
    """
    Monte Carlo Control using Epsilon-Greedy policies.

    Args:
        env: The environment
        num_episodes: Number of episodes to sample
        epsilon: Epsilon value for the epsilon-greedy policy
        discount_factor: Gamma discount factor

    Returns:
        A tuple (Q, policy, stats) where:
            Q is the optimal action-value function
            policy is the optimal greedy policy based on Q
            stats are episode statistics
    """
    # Initialize statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes)
    )

    # Initialize returns and Q-values
    returns_sum = {}
    returns_count = {}
    Q = {}

    # Initialize the policy
    policy = create_epsilon_greedy_policy(Q, epsilon)

    for episode_idx in range(num_episodes):
        # Generate an episode using the current policy
        episode = []
        state = env.reset()

        # Initialize episode
        done = False
        while not done:
            # Get action probabilities
            action_probs = policy(state)
            # Select action based on the probabilities
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # Track episode statistics
        stats.episode_lengths[episode_idx] = len(episode)
        stats.episode_rewards[episode_idx] = sum(reward for _, _, reward in episode)

        # Calculate returns for each step
        G = 0
        # Process states in reverse order
        visited_pairs = set()

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + discount_factor * G

            # Check first-visit condition
            sa_pair = (state, action)
            if sa_pair not in visited_pairs:
                visited_pairs.add(sa_pair)

                # Update return sums and counts
                if sa_pair not in returns_sum:
                    returns_sum[sa_pair] = 0
                    returns_count[sa_pair] = 0

                returns_sum[sa_pair] += G
                returns_count[sa_pair] += 1

                # Update Q-value based on average return
                Q[sa_pair] = returns_sum[sa_pair] / returns_count[sa_pair]

        # Print progress
        if (episode_idx + 1) % 1000 == 0:
            print(f"Completed {episode_idx + 1} episodes (epsilon-greedy MC control)")

            # Add debugging: occasionally print some policy decisions
            if (episode_idx + 1) % 10000 == 0:
                print("Debug - Sample policy decisions:")
                for player_sum in [12, 16, 20]:
                    for dealer_card in [2, 6, 10]:
                        for usable_ace in [0, 1]:
                            test_state = (player_sum, dealer_card, usable_ace)
                            q_stick = Q.get((test_state, 0), 0.0)
                            q_hit = Q.get((test_state, 1), 0.0)
                            best_action = "Stick" if q_stick >= q_hit else "Hit"
                            print(
                                f"  Player: {player_sum}, Dealer: {dealer_card}, Usable Ace: {usable_ace} → {best_action} (Q_stick: {q_stick:.3f}, Q_hit: {q_hit:.3f})"
                            )

    # Create a deterministic greedy policy based on Q-values
    def optimal_policy(state):
        q_values = [Q.get((state, a), 0.0) for a in range(2)]

        # Handle the case of equal Q-values based on common blackjack strategy
        if q_values[0] == q_values[1]:
            # For equal values (including zeros), use a simple heuristic
            return 1 if state[0] < 17 else 0  # hit if < 17
        else:
            return np.argmax(q_values)

    return Q, optimal_policy, stats


def plot_value_function_plotly(V, title, usable_ace=False):
    """Plot the value function as an interactive 3D surface plot using Plotly."""
    # Create grid for player sum and dealer showing card
    player_sums = list(range(12, 22))
    dealer_cards = list(range(1, 11))

    # Initialize grid for value function
    value_grid = np.zeros((len(player_sums), len(dealer_cards)))

    # Fill the grid with values
    for i, player_sum in enumerate(player_sums):
        for j, dealer_card in enumerate(dealer_cards):
            state = (player_sum, dealer_card, int(usable_ace))
            if state in V:
                value_grid[i, j] = V[state]

    # Create meshgrid for plotting
    X, Y = np.meshgrid(dealer_cards, player_sums)

    # Create 3D surface plot with Plotly
    fig = go.Figure(
        data=[
            go.Surface(
                z=value_grid,
                x=dealer_cards,
                y=player_sums,
                colorscale="Viridis",
                showscale=True,
                opacity=0.9,
            )
        ]
    )

    # Update layout
    fig.update_layout(
        title=f"{title} (Usable Ace: {usable_ace})",
        scene=dict(
            xaxis_title="Dealer Showing",
            yaxis_title="Player Sum",
            zaxis_title="Value",
            xaxis=dict(nticks=10, range=[1, 10]),
            yaxis=dict(nticks=10, range=[12, 21]),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        width=800,
        height=600,
        margin=dict(l=20, r=20, b=20, t=60),
    )

    return fig


def plot_comparison_plotly(V_first, V_every):
    """Plot the difference between first-visit and every-visit MC using Plotly."""
    # Create grid for player sum and dealer showing card
    player_sums = list(range(12, 22))
    dealer_cards = list(range(1, 11))

    # Initialize grid for value function differences
    diff_grid_no_ace = np.zeros((len(player_sums), len(dealer_cards)))
    diff_grid_ace = np.zeros((len(player_sums), len(dealer_cards)))

    # Fill the grid with differences
    for i, player_sum in enumerate(player_sums):
        for j, dealer_card in enumerate(dealer_cards):
            # No usable ace
            state_no_ace = (player_sum, dealer_card, 0)
            if state_no_ace in V_first and state_no_ace in V_every:
                diff_grid_no_ace[i, j] = V_first[state_no_ace] - V_every[state_no_ace]

            # Usable ace
            state_ace = (player_sum, dealer_card, 1)
            if state_ace in V_first and state_ace in V_every:
                diff_grid_ace[i, j] = V_first[state_ace] - V_every[state_ace]

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "First-Visit - Every-Visit (No Usable Ace)",
            "First-Visit - Every-Visit (Usable Ace)",
        ),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]],
    )

    # Add heatmaps
    fig.add_trace(
        go.Heatmap(
            z=diff_grid_no_ace,
            x=dealer_cards,
            y=player_sums,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Difference", x=-0.15),
            showscale=True,
            hovertemplate="Player: %{y}<br>Dealer: %{x}<br>Diff: %{z:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=diff_grid_ace,
            x=dealer_cards,
            y=player_sums,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Difference", x=1.1),
            showscale=True,
            hovertemplate="Player: %{y}<br>Dealer: %{x}<br>Diff: %{z:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title_text="Difference Between First-Visit and Every-Visit MC",
        xaxis=dict(title="Dealer Showing"),
        xaxis2=dict(title="Dealer Showing"),
        yaxis=dict(title="Player Sum"),
        yaxis2=dict(title="Player Sum"),
        height=500,
        width=1000,
    )

    return fig


def plot_policy_visualization(V_first, V_every):
    """Create visualizations comparing state visitation frequency."""
    # Create a DataFrame for easy plotting
    import pandas as pd

    data = []
    # Process values for states with and without usable ace
    for usable_ace in [0, 1]:
        ace_label = "Usable Ace" if usable_ace == 1 else "No Usable Ace"
        for player_sum in range(12, 22):
            for dealer_card in range(1, 11):
                state = (player_sum, dealer_card, usable_ace)

                # Add first-visit value if state exists
                if state in V_first:
                    data.append(
                        {
                            "Player Sum": player_sum,
                            "Dealer Showing": dealer_card,
                            "Ace": ace_label,
                            "Method": "First-Visit",
                            "Value": V_first[state],
                        }
                    )

                # Add every-visit value if state exists
                if state in V_every:
                    data.append(
                        {
                            "Player Sum": player_sum,
                            "Dealer Showing": dealer_card,
                            "Ace": ace_label,
                            "Method": "Every-Visit",
                            "Value": V_every[state],
                        }
                    )

    df = pd.DataFrame(data)

    # Create a scatter plot with 3D matrix for both methods
    fig = px.scatter_3d(
        df,
        x="Dealer Showing",
        y="Player Sum",
        z="Value",
        color="Method",
        symbol="Ace",
        opacity=0.7,
        color_discrete_map={"First-Visit": "blue", "Every-Visit": "red"},
        title="Comparison of First-Visit vs Every-Visit MC",
        labels={"Value": "State Value"},
    )

    # Improve layout
    fig.update_layout(
        scene=dict(
            xaxis_title="Dealer Showing",
            yaxis_title="Player Sum",
            zaxis_title="Value",
        ),
        legend_title="Method and Ace",
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig


def plot_q_value_function_plotly(Q, title, usable_ace=False):
    """
    Plot the Q-value function as an interactive 3D surface plot using Plotly.
    Create separate plots for action=0 (stick) and action=1 (hit).
    """
    # Create grid for player sum and dealer showing card
    player_sums = list(range(12, 22))
    dealer_cards = list(range(1, 11))

    # Initialize grids for Q-value function, one for each action
    q_grid_stick = np.zeros((len(player_sums), len(dealer_cards)))
    q_grid_hit = np.zeros((len(player_sums), len(dealer_cards)))

    # Fill the grids with values
    for i, player_sum in enumerate(player_sums):
        for j, dealer_card in enumerate(dealer_cards):
            state = (player_sum, dealer_card, int(usable_ace))

            # Get Q-value for stick (action=0)
            sa_pair_stick = (state, 0)
            if sa_pair_stick in Q:
                q_grid_stick[i, j] = Q[sa_pair_stick]

            # Get Q-value for hit (action=1)
            sa_pair_hit = (state, 1)
            if sa_pair_hit in Q:
                q_grid_hit[i, j] = Q[sa_pair_hit]

    # Create subplots - one for each action
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Stick (Action=0)", "Hit (Action=1)"),
        specs=[[{"type": "surface"}, {"type": "surface"}]],
    )

    # Add stick action surface
    fig.add_trace(
        go.Surface(
            z=q_grid_stick,
            x=dealer_cards,
            y=player_sums,
            colorscale="Viridis",
            showscale=True,
            opacity=0.9,
            name="Stick",
        ),
        row=1,
        col=1,
    )

    # Add hit action surface
    fig.add_trace(
        go.Surface(
            z=q_grid_hit,
            x=dealer_cards,
            y=player_sums,
            colorscale="Viridis",
            showscale=True,
            opacity=0.9,
            name="Hit",
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title=f"{title} (Usable Ace: {usable_ace})",
        scene=dict(
            xaxis_title="Dealer Showing",
            yaxis_title="Player Sum",
            zaxis_title="Q-Value",
            xaxis=dict(nticks=10, range=[1, 10]),
            yaxis=dict(nticks=10, range=[12, 21]),
        ),
        scene2=dict(
            xaxis_title="Dealer Showing",
            yaxis_title="Player Sum",
            zaxis_title="Q-Value",
            xaxis=dict(nticks=10, range=[1, 10]),
            yaxis=dict(nticks=10, range=[12, 21]),
        ),
        width=1200,
        height=600,
        margin=dict(l=20, r=20, b=20, t=60),
    )

    return fig


def plot_optimal_policy_from_q(Q, title, usable_ace=False):
    """Plot the optimal policy derived from Q-values using Plotly."""
    # Create grid for player sum and dealer showing card
    player_sums = list(range(12, 22))
    dealer_cards = list(range(1, 11))

    # Initialize grid for optimal policy
    policy_grid = np.zeros((len(player_sums), len(dealer_cards)))

    # Initialize grid to track which states have valid Q-values
    q_diff_grid = np.zeros((len(player_sums), len(dealer_cards)))

    # Fill the grid with optimal actions
    for i, player_sum in enumerate(player_sums):
        for j, dealer_card in enumerate(dealer_cards):
            state = (player_sum, dealer_card, int(usable_ace))

            # Get Q-values for both actions
            q_stick = Q.get((state, 0), 0.0)
            q_hit = Q.get((state, 1), 0.0)

            # Store Q-value difference for coloring
            q_diff_grid[i, j] = q_hit - q_stick

            # Choose the best action (1 for hit, 0 for stick)
            # For equal values, use a simple heuristic based on common blackjack strategy
            if q_hit == q_stick:
                policy_grid[i, j] = 1 if player_sum < 17 else 0
            else:
                policy_grid[i, j] = 1 if q_hit > q_stick else 0

    # Create heatmap for policy
    fig = go.Figure(
        data=go.Heatmap(
            z=policy_grid,
            x=dealer_cards,
            y=player_sums,
            colorscale=[[0, "blue"], [1, "red"]],
            showscale=True,
            colorbar=dict(title="Action", tickvals=[0, 1], ticktext=["Stick", "Hit"]),
            hovertemplate="Player Sum: %{y}<br>Dealer Showing: %{x}<br>Action: %{z}<br>Q-diff: %{customdata:.4f}<extra></extra>",
            customdata=q_diff_grid,
        )
    )

    # Update layout
    fig.update_layout(
        title=f"{title} (Usable Ace: {usable_ace})",
        xaxis=dict(title="Dealer Showing", nticks=10),
        yaxis=dict(title="Player Sum", nticks=10),
        width=600,
        height=500,
    )

    return fig


def plot_q_value_advantage(Q, title, usable_ace=False):
    """
    Plot the advantage of hitting vs sticking: Q(s,hit) - Q(s,stick)
    This shows how much better/worse hitting is compared to sticking.
    """
    # Create grid for player sum and dealer showing card
    player_sums = list(range(12, 22))
    dealer_cards = list(range(1, 11))

    # Initialize grid for Q-value advantage
    q_advantage_grid = np.zeros((len(player_sums), len(dealer_cards)))

    # Fill the grid with Q-value advantage
    for i, player_sum in enumerate(player_sums):
        for j, dealer_card in enumerate(dealer_cards):
            state = (player_sum, dealer_card, int(usable_ace))

            # Get Q-values for both actions
            q_stick = Q.get((state, 0), 0.0)
            q_hit = Q.get((state, 1), 0.0)

            # Calculate advantage of hitting over sticking
            q_advantage_grid[i, j] = q_hit - q_stick

    # Create heatmap for Q-value advantage
    fig = go.Figure(
        data=go.Heatmap(
            z=q_advantage_grid,
            x=dealer_cards,
            y=player_sums,
            colorscale="RdBu_r",  # Red-Blue scale with red for positive values
            zmid=0,  # Center the color scale at 0
            showscale=True,
            colorbar=dict(title="Q(hit) - Q(stick)"),
            hovertemplate="Player Sum: %{y}<br>Dealer Showing: %{x}<br>Q(hit) - Q(stick): %{z:.4f}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"{title} (Usable Ace: {usable_ace})",
        xaxis=dict(title="Dealer Showing", nticks=10),
        yaxis=dict(title="Player Sum", nticks=10),
        width=600,
        height=500,
    )

    return fig


def plot_episode_stats(stats, title="Learning Performance", smoothing_window=10):
    """
    Plot episode statistics using Plotly.

    Args:
        stats: EpisodeStats namedtuple with episode_lengths and episode_rewards
        title: Title for the plot
        smoothing_window: Window size for smoothing the rewards

    Returns:
        Plotly figure with episode statistics
    """
    # Create DataFrame for easier plotting
    df = pd.DataFrame(
        {
            "Episode": np.arange(len(stats.episode_lengths)),
            "Length": stats.episode_lengths,
            "Reward": stats.episode_rewards,
        }
    )

    # Calculate smoothed rewards
    df["Smoothed_Reward"] = (
        df["Reward"].rolling(window=smoothing_window, min_periods=1).mean()
    )

    # Calculate cumulative time steps
    df["Cumulative_Steps"] = df["Length"].cumsum()

    # Create subplots with 3 rows
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Episode Length over Time",
            "Episode Reward over Time",
            "Episode vs. Time Steps",
        ),
        vertical_spacing=0.1,
    )

    # Add episode length trace
    fig.add_trace(
        go.Scatter(
            x=df["Episode"], y=df["Length"], mode="lines", name="Episode Length"
        ),
        row=1,
        col=1,
    )

    # Add episode reward traces (raw and smoothed)
    fig.add_trace(
        go.Scatter(
            x=df["Episode"],
            y=df["Reward"],
            mode="lines",
            name="Episode Reward",
            line=dict(color="rgba(31, 119, 180, 0.3)"),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Episode"],
            y=df["Smoothed_Reward"],
            mode="lines",
            name=f"Smoothed Reward (window={smoothing_window})",
            line=dict(color="rgba(31, 119, 180, 1.0)", width=2),
        ),
        row=2,
        col=1,
    )

    # Add episodes vs. time steps
    fig.add_trace(
        go.Scatter(
            x=df["Cumulative_Steps"],
            y=df["Episode"],
            mode="lines",
            name="Episodes vs. Time Steps",
        ),
        row=3,
        col=1,
    )

    # Update layout
    fig.update_layout(
        title=title,
        height=900,
        width=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update x and y axis labels
    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_yaxes(title_text="Length", row=1, col=1)

    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_yaxes(title_text="Reward", row=2, col=1)

    fig.update_xaxes(title_text="Time Steps", row=3, col=1)
    fig.update_yaxes(title_text="Episode", row=3, col=1)

    return fig


def plot_learning_curves(
    results_dict, title="Learning Curves", metric="reward", smoothing_window=100
):
    """
    Plot learning curves for different configurations.

    Args:
        results_dict: Dictionary mapping configuration names to their results
        title: Title for the plot
        metric: 'reward' or 'length' to plot
        smoothing_window: Window size for smoothing

    Returns:
        Plotly figure
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly  # Get a color palette

    for i, (config_name, results) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]

        if metric == "reward":
            data = results["stats"]["episode_rewards"]
            y_label = "Average Reward"
        else:
            data = results["stats"]["episode_lengths"]
            y_label = "Episode Length"

        # Calculate smoothed data
        smoothed_data = (
            pd.Series(data).rolling(window=smoothing_window, min_periods=1).mean()
        )

        # Add trace
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(smoothed_data)),
                y=smoothed_data,
                mode="lines",
                name=config_name,
                line=dict(color=color),
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{title} ({y_label})",
        xaxis_title="Episode",
        yaxis_title=y_label,
        height=600,
        width=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def run_policy_evaluation_experiment(
    env, policy_fn, policy_name, episodes=10000, method="first_visit"
):
    """
    Run a policy evaluation experiment with a specific policy.

    Args:
        env: The environment
        policy_fn: Policy function to evaluate
        policy_name: String name of the policy for display
        episodes: Number of episodes to run
        method: 'first_visit' or 'every_visit' Monte Carlo

    Returns:
        Dictionary with results
    """
    print(f"Running {method} Monte Carlo for {policy_name} ({episodes} episodes)...")

    if method == "first_visit":
        V, stats = monte_carlo_first_visit(env, policy_fn, num_episodes=episodes)
    elif method == "every_visit":
        V, stats = monte_carlo_every_visit(env, policy_fn, num_episodes=episodes)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create performance plot
    plot_title = (
        f"{policy_name} - {method.replace('_', ' ').title()} MC ({episodes} episodes)"
    )
    performance_fig = plot_episode_stats(
        stats, title=plot_title, smoothing_window=max(1, episodes // 100)
    )

    # Create value function plots
    v_no_ace_fig = plot_value_function_plotly(V, plot_title, usable_ace=False)
    v_ace_fig = plot_value_function_plotly(V, plot_title, usable_ace=True)

    results = {
        "V": V,
        "stats": stats,
        "figures": {
            "performance": performance_fig,
            "v_no_ace": v_no_ace_fig,
            "v_ace": v_ace_fig,
        },
    }

    return results


def run_epsilon_greedy_experiment(env, epsilon, episodes=10000):
    """
    Run an epsilon-greedy Monte Carlo control experiment.

    Args:
        env: The environment
        epsilon: Epsilon value for exploration
        episodes: Number of episodes to run

    Returns:
        Dictionary with results
    """
    print(
        f"Running epsilon-greedy MC control (epsilon={epsilon}, {episodes} episodes)..."
    )

    Q, optimal_policy, stats = monte_carlo_control_epsilon_greedy(
        env, num_episodes=episodes, epsilon=epsilon
    )

    # Create performance plot
    plot_title = f"Epsilon-Greedy MC Control (ε={epsilon}, {episodes} episodes)"
    performance_fig = plot_episode_stats(
        stats, title=plot_title, smoothing_window=max(1, episodes // 100)
    )

    # Create Q-value and policy plots
    q_no_ace_fig = plot_q_value_function_plotly(Q, plot_title, usable_ace=False)
    q_ace_fig = plot_q_value_function_plotly(Q, plot_title, usable_ace=True)
    policy_no_ace_fig = plot_optimal_policy_from_q(Q, plot_title, usable_ace=False)
    policy_ace_fig = plot_optimal_policy_from_q(Q, plot_title, usable_ace=True)

    # Add Q-value advantage plots
    q_advantage_no_ace_fig = plot_q_value_advantage(Q, plot_title, usable_ace=False)
    q_advantage_ace_fig = plot_q_value_advantage(Q, plot_title, usable_ace=True)

    results = {
        "Q": Q,
        "optimal_policy": optimal_policy,
        "stats": stats,
        "figures": {
            "performance": performance_fig,
            "q_no_ace": q_no_ace_fig,
            "q_ace": q_ace_fig,
            "policy_no_ace": policy_no_ace_fig,
            "policy_ace": policy_ace_fig,
            "q_advantage_no_ace": q_advantage_no_ace_fig,
            "q_advantage_ace": q_advantage_ace_fig,
        },
    }

    return results


def create_question1_dashboard(results):
    """
    Create dashboard for Question 1: Policy that sticks if player sum > 18.
    Only show the most informative plots that answer the question.
    """
    # Extract the results for 500,000 episodes (more accurate)
    result_500k = results.get("Stick if sum > 18 (500000 episodes)")

    if not result_500k:
        return None

    # Create a subplot with two value function plots side by side
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("No Usable Ace", "Usable Ace"),
        specs=[[{"type": "surface"}, {"type": "surface"}]],
    )

    # Extract the surface data from each plot
    for i, surface_fig in enumerate(
        [result_500k["figures"]["v_no_ace"], result_500k["figures"]["v_ace"]]
    ):
        # Extract surface data from figure
        surface_data = surface_fig.data[0]
        fig.add_trace(surface_data, row=1, col=i + 1)

    # Update layout
    fig.update_layout(
        title_text="Value Function for Policy: Stick if sum > 18 (500,000 episodes)",
        height=600,
        width=1200,
        scene=dict(
            xaxis_title="Dealer Showing",
            yaxis_title="Player Sum",
            zaxis_title="Value",
        ),
        scene2=dict(
            xaxis_title="Dealer Showing",
            yaxis_title="Player Sum",
            zaxis_title="Value",
        ),
    )

    return fig


def create_question2_dashboard(results):
    """
    Create dashboard for Question 2: Policy that sticks if player sum > 17,
    otherwise hit with 30% probability.
    """
    # Extract the results for 500,000 episodes (more accurate)
    result_500k = results.get(
        "Stick if sum > 17, else hit with 30% prob (500000 episodes)"
    )

    if not result_500k:
        return None

    # Create a subplot with two value function plots side by side
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("No Usable Ace", "Usable Ace"),
        specs=[[{"type": "surface"}, {"type": "surface"}]],
    )

    # Extract the surface data from each plot
    for i, surface_fig in enumerate(
        [result_500k["figures"]["v_no_ace"], result_500k["figures"]["v_ace"]]
    ):
        # Extract surface data from figure
        surface_data = surface_fig.data[0]
        fig.add_trace(surface_data, row=1, col=i + 1)

    # Update layout
    fig.update_layout(
        title_text="Value Function for Stochastic Policy: Stick if sum > 17, else hit with 30% probability (500,000 episodes)",
        height=600,
        width=1200,
        scene=dict(
            xaxis_title="Dealer Showing",
            yaxis_title="Player Sum",
            zaxis_title="Value",
        ),
        scene2=dict(
            xaxis_title="Dealer Showing",
            yaxis_title="Player Sum",
            zaxis_title="Value",
        ),
    )

    return fig


def create_question3_dashboard(results):
    """
    Create dashboard for Question 3: Compare first-visit and every-visit MC.
    """
    # Get results for 500,000 episodes
    first_visit_results = results.get("Simple Policy - First Visit (500000 episodes)")
    every_visit_results = results.get("Simple Policy - Every Visit (500000 episodes)")

    if not first_visit_results or not every_visit_results:
        return None

    # Create difference heatmaps
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "First-Visit MC (No Usable Ace)",
            "Every-Visit MC (No Usable Ace)",
            "First-Visit MC (Usable Ace)",
            "Every-Visit MC (Usable Ace)",
        ),
        specs=[
            [{"type": "surface"}, {"type": "surface"}],
            [{"type": "surface"}, {"type": "surface"}],
        ],
        vertical_spacing=0.1,
    )

    # Add first visit surfaces
    fig.add_trace(first_visit_results["figures"]["v_no_ace"].data[0], row=1, col=1)
    fig.add_trace(first_visit_results["figures"]["v_ace"].data[0], row=2, col=1)

    # Add every visit surfaces
    fig.add_trace(every_visit_results["figures"]["v_no_ace"].data[0], row=1, col=2)
    fig.add_trace(every_visit_results["figures"]["v_ace"].data[0], row=2, col=2)

    # Create a separate figure for the difference heatmap
    diff_fig = plot_comparison_plotly(
        first_visit_results["V"], every_visit_results["V"]
    )

    # Return both figures as a dictionary
    return {"value_functions": fig, "differences": diff_fig}


def create_question4_dashboard(epsilon_results, episodes=500000):
    """
    Create dashboard for Question 4: Epsilon-greedy MC control with different epsilon values.

    Args:
        epsilon_results: Dictionary mapping epsilon values to results
        episodes: Number of episodes

    Returns:
        Dashboard figure
    """
    # Filter for results with the specified episode count
    filtered_results = {
        k: v for k, v in epsilon_results.items() if f"{episodes} episodes" in k
    }

    if not filtered_results:
        return None

    # Create a dict that maps epsilon values to their results
    epsilon_dict = {}
    for key, results in filtered_results.items():
        # Extract epsilon value from the key
        epsilon = key.split("ε=")[1].split(",")[0]
        epsilon_dict[f"ε={epsilon}"] = results

    # Create learning curves comparison
    reward_comparison_fig = plot_learning_curves(
        {
            name: {"stats": {"episode_rewards": results["stats"].episode_rewards}}
            for name, results in epsilon_dict.items()
        },
        title=f"Epsilon-Greedy MC Control: Learning Curves ({episodes} episodes)",
        metric="reward",
        smoothing_window=max(1, episodes // 100),
    )

    # Create a subplot with optimal policies for each epsilon value
    policy_fig = make_subplots(
        rows=len(epsilon_dict),
        cols=2,
        subplot_titles=[
            f"Optimal Policy ε={eps.split('=')[1]} (No Usable Ace)"
            for eps in epsilon_dict.keys()
        ]
        + [
            f"Optimal Policy ε={eps.split('=')[1]} (Usable Ace)"
            for eps in epsilon_dict.keys()
        ],
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]] * len(epsilon_dict),
        vertical_spacing=0.05,
    )

    # Add policy heatmaps
    for i, (eps_key, results) in enumerate(epsilon_dict.items()):
        # Add policy for no usable ace
        policy_no_ace_trace = results["figures"]["policy_no_ace"].data[0]
        policy_fig.add_trace(policy_no_ace_trace, row=i + 1, col=1)

        # Add policy for usable ace
        policy_ace_trace = results["figures"]["policy_ace"].data[0]
        policy_fig.add_trace(policy_ace_trace, row=i + 1, col=2)

    # Update layout for policy figure
    policy_fig.update_layout(
        title_text=f"Optimal Policies for Different Epsilon Values ({episodes} episodes)",
        height=250 * len(epsilon_dict),
        width=1200,
    )

    # Create a subplot for Q-value advantages (Q_hit - Q_stick)
    advantage_fig = make_subplots(
        rows=len(epsilon_dict),
        cols=2,
        subplot_titles=[
            f"Q-value Advantage ε={eps.split('=')[1]} (No Usable Ace)"
            for eps in epsilon_dict.keys()
        ]
        + [
            f"Q-value Advantage ε={eps.split('=')[1]} (Usable Ace)"
            for eps in epsilon_dict.keys()
        ],
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]] * len(epsilon_dict),
        vertical_spacing=0.05,
    )

    # For each epsilon value
    for i, (eps_key, results) in enumerate(epsilon_dict.items()):
        Q = results["Q"]

        # Create advantage visualization for no usable ace
        q_advantage_no_ace = plot_q_value_advantage(
            Q, f"Q-value Advantage ε={eps_key.split('=')[1]}", usable_ace=False
        )
        advantage_fig.add_trace(q_advantage_no_ace.data[0], row=i + 1, col=1)

        # Create advantage visualization for usable ace
        q_advantage_ace = plot_q_value_advantage(
            Q, f"Q-value Advantage ε={eps_key.split('=')[1]}", usable_ace=True
        )
        advantage_fig.add_trace(q_advantage_ace.data[0], row=i + 1, col=2)

    # Update layout for advantage figure
    advantage_fig.update_layout(
        title_text=f"Q-value Advantage (Q_hit - Q_stick) for Different Epsilon Values ({episodes} episodes)",
        height=250 * len(epsilon_dict),
        width=1200,
    )

    # Return figures as a dictionary
    return {
        "learning_curves": reward_comparison_fig,
        "optimal_policies": policy_fig,
        "q_value_advantage": advantage_fig,
    }


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create environment
    env = BlackjackEnv()

    # Configuration for experiments
    config = {
        # Select experiment(s) to run
        "run_experiments": [
            # "policy_stick_above_18",  # Question 1
            # "policy_stick_above_17_prob_hit",  # Question 2
            "policy_comparison",  # Question 3
            # "epsilon_greedy",  # Question 4
        ],
        # Episode counts for experiments
        "episode_counts": [10000, 500000],
        # Parameters for epsilon-greedy experiments
        "epsilon_values": [0.2, 0.5, 0.9],
        # Show plots interactively (set to False for batch runs)
        "show_plots": True,
        # Save results to files
        "save_results": False,
    }

    all_results = {}
    dashboard_figures = {}

    print("Running experiments for all questions...")

    # Run experiments based on configuration
    for experiment in config["run_experiments"]:
        if experiment == "policy_stick_above_18":
            # Question 1: Policy that sticks if player sum > 18
            for episodes in config["episode_counts"]:
                policy_name = "Stick if sum > 18"
                results = run_policy_evaluation_experiment(
                    env, policy_stick_above_18, policy_name, episodes=episodes
                )
                all_results[f"{policy_name} ({episodes} episodes)"] = results

        elif experiment == "policy_stick_above_17_prob_hit":
            # Question 2: Policy that sticks if player sum > 17, otherwise hit with 30% probability
            for episodes in config["episode_counts"]:
                policy_name = "Stick if sum > 17, else hit with 30% prob"
                results = run_policy_evaluation_experiment(
                    env, policy_stick_above_17_prob_hit, policy_name, episodes=episodes
                )
                all_results[f"{policy_name} ({episodes} episodes)"] = results

        elif experiment == "policy_comparison":
            # Question 3: Compare first-visit and every-visit MC.
            for episodes in config["episode_counts"]:
                # First-visit MC
                first_visit_results = run_policy_evaluation_experiment(
                    env,
                    simple_policy,
                    "Simple Policy",
                    episodes=episodes,
                    method="first_visit",
                )

                # Every-visit MC
                every_visit_results = run_policy_evaluation_experiment(
                    env,
                    simple_policy,
                    "Simple Policy",
                    episodes=episodes,
                    method="every_visit",
                )

                # Add to all results
                all_results[f"Simple Policy - First Visit ({episodes} episodes)"] = (
                    first_visit_results
                )
                all_results[f"Simple Policy - Every Visit ({episodes} episodes)"] = (
                    every_visit_results
                )

        elif experiment == "epsilon_greedy":
            # Question 4: Epsilon-greedy MC control with different epsilon values
            for episodes in config["episode_counts"]:
                for epsilon in config["epsilon_values"]:
                    # Run epsilon-greedy experiment
                    results = run_epsilon_greedy_experiment(
                        env, epsilon, episodes=episodes
                    )
                    # Add to results dictionary
                    all_results[
                        f"Epsilon-Greedy (ε={epsilon}, {episodes} episodes)"
                    ] = results

    print("All experiments completed. Creating dashboards...")

    # Create dashboards for each question
    if "policy_stick_above_18" in config["run_experiments"]:
        dashboard_figures["question1"] = create_question1_dashboard(all_results)

    if "policy_stick_above_17_prob_hit" in config["run_experiments"]:
        dashboard_figures["question2"] = create_question2_dashboard(all_results)

    if "policy_comparison" in config["run_experiments"]:
        dashboard_figures["question3"] = create_question3_dashboard(all_results)

    if "epsilon_greedy" in config["run_experiments"]:
        dashboard_figures["question4"] = create_question4_dashboard(all_results)

    # Display dashboards if configured
    if config["show_plots"]:
        print("\nDisplaying dashboards for all questions...\n")

        for question, figure in dashboard_figures.items():
            if figure:
                print(f"Showing dashboard for {question}...")
                if isinstance(figure, dict):
                    for name, fig in figure.items():
                        print(f"  - {name}")
                        fig.show()
                else:
                    figure.show()

    print("\nAll dashboards displayed. Analysis complete!")
    return all_results, dashboard_figures


if __name__ == "__main__":
    main()
