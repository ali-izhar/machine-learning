# Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) is a policy gradient method for reinforcement learning that alternates between sampling data from the policy and optimizing a "surrogate" objective function using stochastic gradient ascent.

## Algorithm

PPO combines several key ideas:
- Actor-critic architecture with policy and value networks
- Trust region policy optimization with a clipped surrogate objective
- Generalized Advantage Estimation (GAE) for variance reduction
- Mini-batch optimization on collected trajectories

### Mathematical Formulation

**Clipped Surrogate Objective**:
PPO uses a clipped surrogate objective to constrain policy updates:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

where $r_t(\theta)$ is the probability ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

This clipping prevents the new policy from deviating too far from the old policy, promoting more stable learning.

**Value Function Loss**:
The value network is trained to estimate the state value function:

$$L^{VF}(\theta) = \mathbb{E}_t[(V_\theta(s_t) - V_t^{target})^2]$$

**Combined Objective**:
The full PPO objective includes the clipped surrogate objective, value function loss, and an entropy bonus:

$$L^{Total}(\theta) = \mathbb{E}_t[L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t)]$$

where $S$ is the entropy bonus encouraging exploration.

**Advantage Estimation**:
PPO uses Generalized Advantage Estimation (GAE):

$$\hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + ... + (\gamma \lambda)^{T-t+1} \delta_{T-1}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

## Implementation

The repository consists of three main components:
1. `model.py`: Actor and critic network architectures
2. `ppo_agent.py`: PPO agent implementation with advantage estimation and policy updates
3. `train.py`: Environment setup, data collection, and training/evaluation loops

### Network Architecture

**Actor**: Parameterized policy $\pi_\theta(a|s)$
- 3 hidden layers with 256 units each
- Outputs mean and scale parameters for TanhNormal distribution
- Action range constrained via tanh transformation

**Critic**: Value function $V_\theta(s)$
- 3 hidden layers with 256 units each
- Outputs a scalar representing the state value estimate

## Usage

Train a PPO agent:
```bash
python train.py --mode train --env InvertedDoublePendulum-v4 --frames 500000
```

Test a trained agent:
```bash
python train.py --mode test --env InvertedDoublePendulum-v4 --checkpoint results/model_final.pt
```

## Hyperparameters

```
FRAMES_PER_BATCH = 1000    # Number of environment steps per batch
CLIP_EPSILON = 0.2         # PPO clipping parameter
GAMMA = 0.99               # Discount factor
LAMBDA = 0.95              # GAE lambda parameter
ENTROPY_EPS = 1e-4         # Entropy bonus coefficient
NUM_EPOCHS = 10            # Number of optimization epochs per batch
SUB_BATCH_SIZE = 64        # Mini-batch size for updates
LEARNING_RATE = 3e-4       # Learning rate
MAX_GRAD_NORM = 1.0        # Maximum gradient norm for clipping
```

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
2. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438.
3. PyTorch RL Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html 