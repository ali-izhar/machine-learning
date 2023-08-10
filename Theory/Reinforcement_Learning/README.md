# Reinforcement Learning
Reinforcement learning is a type of machine learning that enables an agent to learn in an interactive environment by trial and error using feedback from its own actions and experiences. Reinforcement learning is different from supervised learning in a way that in supervised learning the training data has the answer key with it so the model is trained with the correct answer itself whereas in reinforcement learning, there is no answer but the reinforcement agent decides what to do to perform the given task. In the absence of training dataset, it is bound to learn from its experience.

## Applications
- Controlling robots
- Playing games
- Factory automation
- Self-driving cars
- Financial (stock) trading

## Reinforcement Learning Terminologies
* **Agent**: The learner or decision-maker is called the agent. The agent takes actions (that is, interacts with the environment) and in return, it receives rewards/penalties.
* **Environment**: The world in which the agent is placed is called the environment. The environment is the interface between the agent and the outside world. The agent sends an action to the environment and the environment responds back with an observation and a reward (or penalty).
* **Action**: The set of actions that the agent can perform is called the action space. The action space can be continuous or discrete. For example, the action space for a car can be the steering angle and the acceleration. The action space for a robot can be the joint torques of the robot. The action space for a game can be the set of all possible moves that the player can make.
* **State**: The state of the environment is the set of all the information that describes the environment at a given time. The state can be observable or partially observable. For example, the state of a chess game is the position of all the pieces on the board. The state of a robot can be the joint angles and the joint velocities. The state of a car can be the position and the velocity of the car. The state of a game can be the position of all the players and the ball.
* **Reward**: The reward is the feedback from the environment. The reward can be positive or negative. The goal of the agent is to maximize the total reward. For example, the reward for a chess game can be +1 if the agent wins, 0 if the game is drawn, and -1 if the agent loses. The reward for a car racing game can be +1 if the car crosses the finish line within the given time, 0 if the car fails to cross the finish line within the given time, and -1000, for example, if the car crashes because we want to discourage the car from crashing and thus a large negative reward is given.
* **Policy**: The policy is the strategy that the agent employs to determine the next action based on the current state. The policy can be deterministic or stochastic. For example, the policy for a chess game can be a lookup table that maps the current state to the next action. The policy for a car racing game can be a neural network that takes the current state as input and outputs the steering angle and the acceleration. The policy for a robot can be a set of if-else statements that map the current state to the joint torques.
* **Discount Factor**: The discount factor is a number between 0 and 1 that determines the importance of future rewards. A discount factor of 0 makes the agent short-sighted by only considering current rewards. A discount factor of 1 makes the agent far-sighted by considering future rewards with equal weight as current rewards. For example, if the discount factor is 0.9, then the agent will consider future rewards to be 90% as important as current rewards.
* **Value Function**: The value function is the expected total reward that the agent will receive starting from the current state. The value function is the sum of all the discounted future rewards. The value function can be used to determine the next action. For example, if the value function is high for a given state, then the agent will take an action that leads to that state. The value function can be estimated using a neural network. The value function is also called the Q-function.

In reinforcement learning, the agent is in a state $s_t$ at time $t$. The agent takes an action $a_t$ based on the policy $\pi(s_t) = a_t$. The environment transitions to a new state $s_{t+1}$ and gives a reward $r_{t+1}$. The goal of the agent is to maximize the total reward. The total reward is the sum of all the rewards from time $t=0$ to $t=T$.

## Markov Decision Process
A Markov decision process (MDP) is a discrete-time stochastic control process. It provides a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker. MDPs are useful for studying optimization problems solved via dynamic programming and reinforcement learning.

## State-Action Value Function
The state-action value function $Q(s, a)$ is the expected total reward (or return) that the agent will receive starting from state $s$ and taking action $a$. The state-action value function is also called the Q-function.

$Q(s, a) =$ Return if you start in state $s$, take action $a$ (once), and then behave optimally after that (i.e., follow the optimal policy). The optimal policy is the policy that maximizes the total reward (or return) given the current state: $\max_{a} Q(s, a)$.

$Q(s, a)$ is the expected total reward that the agent will receive starting from state $s$ and taking action $a$ and is also given by:

$$Q(s, a) = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3 R_4 + \cdots$$

$$Q(s, a) = R_1 + \gamma (R_2 + \gamma R_3 + \gamma^2 R_4 + \cdots)$$

$$Q(s, a) = R_1 + \gamma Q(s', a')$$

where $R_1$ is the reward for the current state $s$ (also called the immediate reward), $R_2$ is the reward for the next state $s'$, and so on.

## Bellman Equation
The Bellman equation is a necessary and sufficient condition for optimality associated with the mathematical optimization method known as dynamic programming. It writes the relationship for the value of a decision problem at one point in time, in terms of the payoff from some initial choices and the value of the remaining decision problem that results from those initial choices.

The Bellman equation for the state-action value function $Q(s, a)$ is given by:

$$Q(s, a) = R(s) + \gamma \max_{a'} Q(s', a')$$

where $R(s)$ is the reward for the current state $s$, $\gamma$ is the discount factor, $s'$ is the next state, and $a'$ is the next action. Consider the following illustration with $\gamma = 0.5$:

<img src="media/bellman.png" width="500">

If the agent is in state 2 and takes action $\rightarrow$, then the agent will receive a reward of 0 and will transition to state 3. The return for this action is given by:

$$Q(2, \rightarrow) = R(2) + 0.5 \max_{a'} Q(3, a') = 0 + 0.5 \max(25, 6.25) = 12.5$$

Similarly, if the agent is in state 4 and takes action $\leftarrow$, then the agent will receive a reward of 0 and will transition to state 3. The return for this action is given by:

$$Q(4, \leftarrow) = R(4) + 0.5 \max_{a'} Q(3, a') = 0 + 0.5 \max(25, 6.25) = 12.5$$

## Stochastic Environment
In a stochastic environment, the next state $s'$ and the reward $R(s)$ are random variables. For example, if the agent is in state 2 and takes action $\rightarrow$, then the agent will transition to state 3 with probability 0.8 and will transition to state 1 with probability 0.2 (due to factors like wind, friction, etc.). In this case, the return is a random variable and therefore, the state-action value function $Q(s, a)$ is also a random variable. Therefore, instead of maximizing the state-action value function $Q(s, a)$, we maximize the expected (or average) state-action value function $\mathbb{E}[Q(s, a)]$ given by:

$$Q(s, a) = R(s) + \gamma \max_{a'} \mathbb{E}[Q(s', a')]$$

where $\mathbb{E}[R(s)]$ is the expected reward for the current state $s$, $\gamma$ is the discount factor, $s'$ is the next state, and $a'$ is the next action.