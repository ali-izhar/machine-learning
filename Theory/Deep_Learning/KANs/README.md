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

The **Kolmogorov-Arnold theorem** states that:

> ![Note]
> Any multivariate continuous function can be broken down into a combination of single-variable functions and addition.

Unlike traditional MLPs that use fixed activation functions at each neuron, KANs apply learnable activation functions on the connections (edges) between neurons. These activation functions are modeled as **splines**, allowing the network to adapt and find the best transformations during training.

A **spline** is a piecewise polynomial function that is smooth at the points where the polynomial pieces connect, known as **knots**. Splines can approximate complex functions by stitching together simple polynomial segments, ensuring continuity and smoothness across the entire range. Mathematically, a spline $S(x)$ of degree $n$ is defined as:

$$
S(x) = \begin{cases}
P_1(x) & \text{for } x \in [x_0, x_1] \\
P_2(x) & \text{for } x \in [x_1, x_2] \\
\vdots \\
P_m(x) & \text{for } x \in [x_{m-1}, x_m]
\end{cases}
$$

where each $P_i(x)$ is a polynomial of degree $n$ defined on the interval $[x_{i-1}, x_i]$, and the spline is smooth at each knot $x_i$.

```python
python draw.py -spline
```

**B-splines** (Basis splines) are a specific type of spline that provides greater numerical stability and local control over the shape of the function. They are defined by their degree and a set of control points, which determine the spline's form.

A B-spline $B_i^n(x)$ of degree $n$ is defined as:

$$
B_i^n(x) = \frac{x - x_i}{x_{i+n} - x_i} B_{i}^{n-1}(x) + \frac{x_{i+n+1} - x}{x_{i+n+1} - x_{i+1}} B_{i+1}^{n-1}(x)
$$

where $B_i^0(x) = 1$ if $x_i \leq x \leq x_{i+1}$ and $0$ otherwise.

## References

- The Math Behind KAN - [Medium](https://towardsdatascience.com/the-math-behind-kan-kolmogorov-arnold-networks-7c12a164ba95)
