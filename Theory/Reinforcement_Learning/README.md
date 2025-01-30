# Reinforcement Learning

**Reinforcement Learning (RL)** is a subfield of machine learning where an agent learns to make decisions by interacting with an environment to maximize cumulative rewards. Think of it like training a dog - the dog (agent) learns which behaviors lead to treats (rewards) through trial and error.

## Key Elements of RL

1. **Agent**: Entity that learns a policy $\pi$ to map states ($s \in \mathcal{S}$) to actions ($a \in \mathcal{A}$). Like a player in a video game.
2. **Environment**: Dynamics governing state transitions $s_{t+1} \sim P(s_{t+1} | s_t, a_t)$ and rewards $r_t \sim R(r_t | s_t, a_t)$. Like the game world and its rules.
3. **State** ($s$): Representation of the environment at time $t$. Everything the agent can observe.
4. **Action** ($a$): Decision taken by the agent. What the agent can do.
5. **Reward** ($r$): Scalar feedback signal. Immediate feedback on how good/bad an action was.
6. **Policy** ($\pi$): Strategy $\pi(a|s)$ (stochastic) or $a = \pi(s)$ (deterministic). The agent's decision-making rules.
7. **Value Function** $\big(V^\pi(s), Q^\pi(s,a)\big)$: Expected cumulative reward from state $s$ (and action $a$). Like estimating how good a chess position is.
8. **Model** (Optional): Agent's internal representation of $P(s_{t+1}|s_t, a_t)$ and $R(r_t|s_t, a_t)$. The agent's understanding of how the world works.

## Markov Decision Processes (MDPs)

An MDP is defined by the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:  
- **States**: $\mathcal{S}$ (finite or continuous).  
- **Actions**: $\mathcal{A}$.  
- **Transition Function**: $P(s_{t+1} | s_t, a_t)$.  
- **Reward Function**: $R(s_t, a_t) = \mathbb{E}[r_t | s_t, a_t]$.  
- **Discount Factor**: $\gamma \in [0, 1]$.  

### Objective
Maximize the **expected return** $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$.

## Core Concepts

### 1. Policy
- **Stochastic Policy**: $\pi(a|s) = \mathbb{P}[a_t=a | s_t=s]$.  
- **Deterministic Policy**: $a = \pi(s)$.  

### 2. Value Functions
- **State-Value Function**:  
  $$V^\pi(s) = \mathbb{E}_\pi\left[ G_t | s_t = s \right]$$  
  Think of this as "How good is it to be in this state?"
  Satisfies the **Bellman Equation**:  
  $$V^\pi(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]$$  

- **Action-Value Function**:  
  $$Q^\pi(s,a) = \mathbb{E}_\pi\left[ G_t | s_t = s, a_t = a \right]$$  
  Think of this as "How good is it to take this action in this state?"
  Bellman Equation:  
  $$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \sum_{a'} \pi(a'|s') Q^\pi(s',a')$$  

### 3. Optimality
- **Optimal Value Functions**:  
  $$V^*(s) = \max_\pi V^\pi(s), \quad Q^*(s,a) = \max_\pi Q^\pi(s,a)$$  
- **Bellman Optimality Equations**:  
  $$V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s') \right]$$  
  $$Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s',a')$$  

## Algorithms

### 1. Dynamic Programming (Model-Based)
- **Policy Evaluation**: Iteratively compute $V^\pi$ using Bellman equations. Like planning ahead in chess.
- **Policy Improvement**: Update $\pi$ greedily w.r.t. $Q^\pi$. Choose better moves based on evaluation.
- **Policy Iteration**: Alternate evaluation and improvement. Keep refining strategy.

### 2. Monte Carlo (Model-Free)
- Estimate $V^\pi(s)$ or $Q^\pi(s,a)$ via empirical average of returns. Learn from complete episodes of experience.
- **Update Rule**:  
  $$V(s_t) \leftarrow V(s_t) + \alpha \left[ G_t - V(s_t) \right]$$  
  Adjust estimates based on actual outcomes.

### 3. Temporal Difference (TD) Learning
- **TD(0) Update**:  
  $$V(s_t) \leftarrow V(s_t) + \alpha \left[ r_t + \gamma V(s_{t+1}) - V(s_t) \right]$$  
- **SARSA (On-Policy TD)**:  
  $$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \left[ r_t + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t) \right]$$  

### 4. Q-Learning (Off-Policy TD)
- **Update Rule**:  
  $$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1},a') - Q(s_t,a_t) \right]$$  

### 5. Deep Q-Networks (DQN)
- Approximate $Q^*(s,a)$ with a neural network $Q_\theta(s,a)$.  
- **Loss Function**:  
  $$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a) \right)^2 \right]$$  
  where $\theta^-$ are target network parameters and $\mathcal{D}$ is a replay buffer.  

### 6. Policy Gradient Methods
- Directly optimize $\pi_\theta(a|s)$ using gradient ascent on $\mathbb{E}[G_t]$.  
- **REINFORCE Gradient**:  
  $$\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ G_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right]$$  
- **Actor-Critic**: Combine policy gradient with a value function baseline $V_w(s)$:  
  $$\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ \left( Q_w(s_t,a_t) - V_w(s_t) \right) \nabla_\theta \log \pi_\theta(a_t|s_t) \right]$$  

## Exploration vs. Exploitation

1. **$\epsilon$-Greedy**: Choose random action with probability $\epsilon$, else $a = \arg\max_a Q(s,a)$. Like occasionally trying a new restaurant instead of going to your favorite.
2. **Softmax (Boltzmann)**:  
   $$\pi(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_{a'} e^{Q(s,a')/\tau}}$$  
   with temperature $\tau > 0$. Higher temperature means more random exploration.
3. **Upper Confidence Bound (UCB)**:  
   $$a_t = \arg\max_a \left[ Q(s,a) + c \sqrt{\frac{\ln t}{N(s,a)}} \right]$$  
   Balances trying actions that seem good with those that haven't been tried much.

## Implementation Considerations

- **Computational Complexity**: DP and exact methods scale poorly with $|\mathcal{S}|$ and $|\mathcal{A}|$.  
- **Sample Efficiency**: TD methods > Monte Carlo; experience replay in DQN.  
- **Convergence**: Q-Learning converges to $Q^*$ under Robbins-Monro conditions.  
- **Hyperparameters**: Tune $\alpha$, $\gamma$, $\epsilon$, network architecture, and batch size.  

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*.  
2. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.  
3. Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv.  
4. Silver, D. (2015). *Lectures on Reinforcement Learning*. UCL.  
