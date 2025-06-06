# Deep Deterministic Policy Gradient (DDPG)

DDPG is an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces.

## Algorithm

DDPG combines DQN and policy gradient methods, using:
- Actor network μ(s|θ^μ) that deterministically maps states to actions
- Critic network Q(s,a|θ^Q) that estimates action-value function
- Experience replay and target networks for stability
- Ornstein-Uhlenbeck process for exploration

### Mathematical Formulation

**Policy Gradient Update**:
The actor is updated by applying the chain rule to the expected return J with respect to actor parameters θ^μ:

$$\nabla_{\theta^\mu} J \approx \mathbb{E}_s[\nabla_{\theta^\mu} Q(s,a|\theta^Q)|_{a=\mu(s|\theta^\mu)}]$$
$$= \mathbb{E}_s[\nabla_a Q(s,a|\theta^Q)|_{a=\mu(s|\theta^\mu)} \cdot \nabla_{\theta^\mu} \mu(s|\theta^\mu)]$$

**Critic Update**:
The critic is updated using temporal difference learning by minimizing the loss:

$$L(\theta^Q) = \mathbb{E}_{s,a,r,s'}[(Q(s,a|\theta^Q) - y)^2]$$

where target y is:

$$y = r + \gamma \cdot Q'(s',\mu'(s'|\theta^{\mu'})|\theta^{Q'}) \cdot (1-\text{done})$$

**Target Networks**:
Target networks are updated using Polyak averaging:

$$\theta' \leftarrow \tau\theta + (1-\tau)\theta'$$

where τ ≪ 1 is a smoothing parameter.

## Implementation

The repository consists of three main components:
1. `model.py`: Actor and Critic network architectures
2. `ddpg_agent.py`: Agent implementation with replay buffer and noise process
3. `train.py`: Training and evaluation scripts

### Network Architecture

**Actor**: Deterministic policy μ(s) → a
- Input layer: state_size
- Hidden layers: 400 and 300 units with ReLU
- Output layer: action_size with tanh activation

**Critic**: Action-value function Q(s,a) → ℝ
- State input → 400 units with ReLU
- Combined with action input at second layer
- Second hidden layer: 300 units with ReLU
- Output: scalar Q-value

## Usage

Train a DDPG agent:
```python
python train.py
```

Test a trained agent:
```python
# Using default saved model
python -c "from train import test; test()"

# Using custom model paths
python -c "from train import test; test(actor_path='path/to/actor.pth', critic_path='path/to/critic.pth')"
```

## Hyperparameters

```
BUFFER_SIZE = 100000   # Replay buffer size
BATCH_SIZE = 128       # Minibatch size
GAMMA = 0.99           # Discount factor
TAU = 0.001            # Target network soft update rate
LR_ACTOR = 0.0001      # Actor learning rate
LR_CRITIC = 0.001      # Critic learning rate
```

## References

1. Lillicrap, T. P., et al. (2015). Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971.
2. Silver, D., et al. (2014). Deterministic policy gradient algorithms. ICML.
3. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
