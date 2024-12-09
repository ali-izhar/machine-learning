# Kolmogorov-Arnold Network (KAN)

In this repository, we will use KAN to model time-series data.

## 1. Limitations of MLPs

Multi-Layer Perceptrons (MLPs) consist of layers of connected nodes that learn to represent complex, non-linear patterns by training on data. Each neuron applies a specific **activation function** to the weighted input it receives, passing the transformed data through multiple layers to produce the final output.

- **Fixed Activation Functions**: Every neuron in an MLP uses the same predetermined activation functions, such as `ReLU` or `Sigmoid`. While these functions work well in many cases, they limit the network's ability to adapt and be flexible. This makes it harder for MLPs to optimize for certain tasks or handle specific types of data effectively.
- **Difficulty in Understanding**: MLPs are often seen as "black boxes" because their internal decision-making processes are not easy to interpret. As MLPs become more complex, it becomes increasingly challenging to understand how they make decisions. The combination of fixed activation functions and complex connections between neurons makes it tough to trust and explain the model's predictions without detailed analysis.

## 2. Kolmogorov-Arnold Networks (KANs)

<div align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*imne0qCLt5xlJUnsn3m9pA.png" alt="KAN" width="90%">
</div>



## References

- The Math Behind KAN - [Medium](https://towardsdatascience.com/the-math-behind-kan-kolmogorov-arnold-networks-7c12a164ba95)
