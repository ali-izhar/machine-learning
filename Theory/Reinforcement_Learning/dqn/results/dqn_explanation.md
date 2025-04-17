# Understanding DQN: Experience Replay and Fixed Target Q-Network

## Experience Replay

Experience replay in DQN stores and reuses past experiences to break correlations in the training data and improve sample efficiency.

1. **Storage:** The agent stores transitions (state, action, next_state, reward) in a replay buffer as it interacts with the environment.

2. **Random sampling:** During training, instead of learning from consecutive experiences, the agent randomly samples batches from this buffer, breaking the correlation between consecutive samples.

3. **Benefits:**
   - Breaks harmful correlations in the observation sequence
   - Allows experiences to be used multiple times for learning
   - Reduces variance in updates, leading to more stable learning
   - Better data efficiency as important transitions can be revisited

## Fixed Target Q-Network

The fixed target Q-network in DQN addresses the moving target problem in Q-learning:

1. **Two networks:** DQN maintains two separate networks:
   - Policy Network (online): Used for selecting actions and being updated
   - Target Network: Used for generating target Q-values during training

2. **Delayed updates:** The target network is updated less frequently than the policy network, either through:
   - Periodic hard updates: Copying the policy network weights every N steps
   - Soft updates: Slowly blending the policy network into the target network (θ_target = τ*θ_policy + (1-τ)*θ_target)

3. **Benefits:**
   - Stabilizes training by reducing correlations between the current Q-values and target Q-values
   - Prevents the "moving target" problem where the network chases its own bootstrapped estimates
   - Reduces oscillations and divergence in the training process

Together, these two mechanisms significantly improve the stability and performance of DQN.
