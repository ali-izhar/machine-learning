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

### Finding State Values

To find the value of a state, we theoretically need to follow all possible paths in the state transition tree. However, this becomes practically infeasible as:
- The tree can grow infinitely due to recursion
- States can transition back to previously visited states
- The number of possible paths grows exponentially

This is where the Bellman equation becomes crucial. Instead of computing infinite paths, it breaks down the value into two components:
1. Immediate reward from the current state-action
2. Expected discounted return from the successor state

This recursive definition allows us to compute values efficiently:
$$V(s) = R(s,a) + \gamma \sum_{s'} P(s'|s,a)V(s')$$

### Matrix Form of Bellman Equation

The Bellman equation can be expressed in matrix form as:
$$\vec{V} = \vec{R} + \gamma \vec{P}\vec{V}$$

Where:
- $\vec{V}$ is the vector of state values
- $\vec{R}$ is the vector of rewards
- $\vec{P}$ is the state transition probability matrix
- $\gamma$ is the discount factor

#### Example: Consider a student MDP with states: {Facebook, Class1, Class2, Class3, Sleep, Bar, Pass}

The reward vector $\vec{R}$ and state values $\vec{V}$ are:

$\vec{R} = \begin{bmatrix} -1 \\ -2 \\ -2 \\ -2 \\ 0 \\ 1 \\ 10 \end{bmatrix}$
$\vec{V} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \\ v_4 \\ v_5 \\ v_6 \\ v_7 \end{bmatrix}$

The transition matrix $\vec{P}$ represents the probability of moving from one state to another:

$\vec{P} = \begin{bmatrix}
0.9 & 0.1 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
0.5 & 0.0 & 0.5 & 0.0 & 0.0 & 0.0 & 0.0 \\
0.0 & 0.0 & 0.0 & 0.8 & 0.2 & 0.0 & 0.0 \\
0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.4 & 0.6 \\
0.0 & 0.0 & 0.0 & 0.0 & 1.0 & 0.0 & 0.0 \\
0.0 & 0.2 & 0.4 & 0.4 & 0.0 & 0.0 & 0.0 \\
0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 1.0
\end{bmatrix}$

To solve for the state values, we use:
$$(I - \gamma P)\vec{V} = \vec{R}$$
$$\vec{V} = (I - \gamma P)^{-1}\vec{R}$$

For example, with $\gamma = 0.9$, solving this system gives us the value of each state:
- Facebook: $v_1 = -8.6$
- Class1: $v_2 = -7.2$
- Class2: $v_3 = -4.1$
- Class3: $v_4 = 2.8$
- Sleep: $v_5 = 0.0$
- Bar: $v_6 = -2.3$
- Pass: $v_7 = 10.0$

This shows that:
1. Passing has the highest value (10.0)
2. Being in Facebook has the lowest value (-8.6)
3. Progress through classes (Class1 → Class2 → Class3) shows increasing values
4. Sleep and Bar states have intermediate values

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

## Exploration vs. Exploitation

### Multi-Armed Bandit Strategies

#### 1. Exploration-Only Strategy
In this strategy, the agent randomly selects actions with equal probability regardless of past outcomes.

- **Action Selection**: For each trial $t$, select action $a_t$ uniformly:
  $$P(a_t = i) = \frac{1}{|\mathcal{A}|}, \quad \forall i \in \mathcal{A}$$

- **Expected Return**: For $T$ trials with reward $r$:
  $$\mathbb{E}[R_T] = \sum_{i=1}^{|\mathcal{A}|} \frac{T}{|\mathcal{A}|} \cdot p_i \cdot r$$
  where $p_i$ is the success probability of action $i$

- **Regret**: Difference between optimal and actual return:
  $$\mathcal{L}_{\text{exploration}} = T \cdot \max_i(p_i) \cdot r - \mathbb{E}[R_T]$$

#### 2. Exploitation-Only Strategy
This strategy tests actions sequentially until finding a success, then exploits that action for remaining trials.

- **Action Selection**: For trial $t$:
  $$a_t = \begin{cases}
  i \text{ where } i = (t \bmod |\mathcal{A}|) & \text{if no success yet} \\
  a_{\text{successful}} & \text{if previous success}
  \end{cases}$$

- **Theoretical Return**: For trial $k$:
  $$R_k = P(\text{first success at }k) \cdot \mathbb{E}[\text{future rewards}|k]$$
  where:
  $$P(\text{first success at }k) = p_k \prod_{i=1}^{|\mathcal{A}|} (1-p_i)^{q_k + a_i}$$
  $$\mathbb{E}[\text{future rewards}|k] = 1 + (T-k)p_k$$
  
  Here:
  - $q_k = \lfloor (k-1)/|\mathcal{A}| \rfloor$ (complete cycles)
  - $a_i$ are adjustment factors based on remainder
  - $p_k$ is the success probability of the action tried at trial $k$

- **Total Expected Return**:
  $$\mathbb{E}[R_T] = \sum_{k=1}^T R_k$$

- **Regret**: Similar to exploration-only:
  $$\mathcal{L}_{\text{exploitation}} = T \cdot \max_i(p_i) \cdot r - \mathbb{E}[R_T]$$

1. **$\epsilon$-Greedy**: Choose random action with probability $\epsilon$, else $a = \arg\max_a Q(s,a)$. Like occasionally trying a new restaurant instead of going to your favorite.
2. **Softmax (Boltzmann)**:  
   $$\pi(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_{a'} e^{Q(s,a')/\tau}}$$  
   with temperature $\tau > 0$. Higher temperature means more random exploration.

#### 3. Epsilon-Greedy Strategy
This strategy combines exploration and exploitation using a probability parameter $\epsilon$:

- **Action Selection**:
  $$a_t = \begin{cases}
  \text{random action} & \text{with probability } \epsilon \\
  \arg\max_a Q(a) & \text{with probability } 1-\epsilon
  \end{cases}$$

- **Value Estimation**: For each action $a$, maintain:
  - Action count: $N(a)$ = number of times action $a$ was chosen
  - Reward history: $R(a,t)$ = reward received for action $a$ at trial $t$
  - Q-value estimate at trial $k$:
    $$Q_k(a) = \frac{1}{N(a)} \sum_{t=1}^k R(a,t)$$

- **Update Rules**:
  - On success at trial $k$:
    $$Q_k(a) = \frac{r + \sum_{t=1}^{k-1} R(a,t)}{N(a)}$$
  - On failure at trial $k$:
    $$Q_k(a) = \frac{\sum_{t=1}^{k-1} R(a,t)}{N(a)}$$
  where $r$ is the reward value (typically 1.0)

- **Tie Breaking**: When multiple actions share maximum Q-value:
  $$P(a_t = a | a \in \arg\max_a Q(a)) = \frac{1}{|\arg\max_a Q(a)|}$$

- **Expected Return**: For $T$ trials:
  $$\mathbb{E}[R_T] = T \cdot \left(\epsilon \cdot \sum_{i=1}^{|\mathcal{A}|} \frac{p_i}{|\mathcal{A}|} + (1-\epsilon) \cdot \max_i p_i\right) \cdot r$$
  where $p_i$ is the success probability of action $i$

- **Regret**: Similar to other strategies:
  $$\mathcal{L}_{\text{ε-greedy}} = T \cdot \max_i(p_i) \cdot r - \mathbb{E}[R_T]$$

This strategy typically achieves better performance than pure exploration or exploitation
by balancing between them. The parameter $\epsilon$ controls this trade-off:
- Higher $\epsilon$: More exploration, slower convergence but better chance of finding optimal action
- Lower $\epsilon$: More exploitation, faster convergence but risk of suboptimal action selection

#### 4. Upper Confidence Bound (UCB) Strategy
This strategy addresses a key limitation of ε-greedy: all non-greedy actions are treated equally during exploration.
UCB instead uses uncertainty in value estimates to guide exploration.

**Intuition**:
Imagine you're a doctor testing different treatments. For each treatment, you maintain:
- The average success rate so far (Q(a))
- How confident you are in this estimate (the uncertainty bonus)

The less you've tried a treatment, the less confident you are about its true effectiveness.
UCB adds an "optimism bonus" to less-tried actions, following the principle:
"Be optimistic in the face of uncertainty."

**Hoeffding's Inequality**:
- Starts with the basic form: $\mathcal{P}(\mathbb{E}[X] > \bar{X}_t + u) \leq e^{-2tu^2}$
- To apply to our bandit problem, we replace:
  - $\mathbb{E}[X]$ with $Q(a)$ (true action-value)
  - $\bar{X}_t$ with $\hat{Q}_t(a)$ (estimated action-value at t)
  - $u$ with $U_t(a)$ (upper confidence bound)

This gives us:
$$\mathcal{P}(Q(a) > \hat{Q}_t(a) + U_t(a)) \leq e^{-2tU_t(a)^2}$$

The inequality $Q(a) \leq \hat{Q}_t(a) + U_t(a)$ is crucial because:
- If true value > optimistic estimate: we're being too pessimistic
- If true value ≤ optimistic estimate: we have a valid upper bound

This ensures we:
- Don't underestimate potentially good actions
- Maintain realistic but optimistic estimates
- Gradually tighten bounds as we gather more data

Since we want $Q(a) \leq \hat{Q}_t(a) + U_t(a)$ with high probability:
1. We need $e^{-2tU_t(a)^2}$ to be very small
2. Using $t \geq N_t(a)$ (current trial ≥ times action chosen):
   $$e^{-2tU_t(a)^2} \leq e^{-2N_t(a)U_t(a)^2} = p$$
3. Solving for $U_t(a)$ with $p = t^{-4}$ gives us:
   $$U_t(a) = \sqrt{\frac{2\ln t}{N_t(a)}}$$

- The UCB formula $c\sqrt{\frac{\ln t}{N(a)}}$ comes from this inequality:

- **Action Selection**: Choose action that maximizes UCB value:
  $$a_t = \arg\max_a \left[ Q(a) + c\sqrt{\frac{\ln t}{N(a)}} \right]$$

- **Components**:
  - $Q(a)$: Estimated value of action (exploitation term)
  - $c\sqrt{\frac{\ln t}{N(a)}}$: Uncertainty bonus (exploration term)
    - $c$: Exploration parameter controlling confidence level
    - $\ln t$: Natural log of total trials (grows slowly)
    - $N(a)$: Number of times action $a$ was chosen

- **Key Properties**:
  - Actions with high estimated values are favored (exploitation)
  - Actions with few attempts have high uncertainty bonus (exploration)
  - Uncertainty bonus decreases as actions are tried more
  - No random exploration needed (unlike ε-greedy)

- **Theoretical Advantages**:
  - Provides theoretical guarantees on regret bounds
  - Automatically reduces exploration over time
  - Focuses exploration on promising actions
  - More efficient than ε-greedy in many scenarios

- **Expected Regret**: Grows logarithmically with time:
  $$\mathbb{E}[\mathcal{L}_{\text{UCB}}] = O(\ln T)$$
  Much better than ε-greedy's linear regret growth

#### 5. Thompson Sampling Strategy
This strategy uses Bayesian inference to balance exploration and exploitation by
maintaining probability distributions over each action's true reward probability.

**Intuition**:
Imagine you're a doctor with different treatments:
- For each treatment, maintain a belief about its success rate
- More uncertain treatments have wider distributions
- More certain treatments have narrower distributions
- Each day:
  1. Sample one possible success rate from each treatment's distribution
  2. Use the treatment with highest sampled rate
  3. Update your belief based on the outcome

**Mathematical Formulation**:

1. **Prior Distribution**:
   - For each action $a$, maintain Beta distribution $Beta(\alpha_a, \beta_a)$
   - Initially: $\alpha_a = 1, \beta_a = 1$ (uniform distribution)
   - Mean of Beta: $\mathbb{E}[p_a] = \frac{\alpha_a}{\alpha_a + \beta_a}$

2. **Action Selection**:
   - Sample $\theta_a \sim Beta(\alpha_a, \beta_a)$ for each action
   - Choose action: $a_t = \arg\max_a \theta_a$

3. **Posterior Update**:
   After observing reward $r_t$:
   $$(\alpha_a, \beta_a) \leftarrow \begin{cases}
   (\alpha_a + 1, \beta_a) & \text{if } r_t = 1 \text{ and } a = a_t \\
   (\alpha_a, \beta_a + 1) & \text{if } r_t = 0 \text{ and } a = a_t \\
   (\alpha_a, \beta_a) & \text{if } a \neq a_t
   \end{cases}$$

**Key Properties**:
- Automatically balances exploration/exploitation through uncertainty
- More exploration when uncertain (wide distributions)
- More exploitation when confident (narrow distributions)
- Asymptotically optimal in many scenarios

**Alternative Formulation**:
For non-binary rewards, can use Normal distribution:
- Maintain $\mathcal{N}(\mu_a, \sigma^2/n_a)$ for each action
- $\mu_a$: Sample mean of rewards
- $n_a$: Number of times action chosen
- $\sigma^2$: Reward variance (fixed)

**Expected Regret**:
- Grows logarithmically: $\mathbb{E}[\mathcal{L}_{\text{TS}}] = O(\ln T)$
- Matches theoretical lower bound for many bandit problems
- Often outperforms UCB empirically

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

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*.  
2. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.  
3. Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv.  
4. Silver, D. (2015). *Lectures on Reinforcement Learning*. UCL.  
