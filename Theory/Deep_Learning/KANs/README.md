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

## 3. Understanding Splines

### 3.1 Piecewise Polynomial Splines

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

### 3.2 B-splines: A More Flexible Approach

**B-splines** (Basis splines) provide a more powerful and numerically stable way to represent smooth curves. They have several key components:

1. **Control Points**: These points guide the shape of the curve but don't necessarily lie on it. Think of them as "magnetic points" that pull the curve in their direction. Moving a control point affects the curve locally, which makes B-splines ideal for local shape control.

2. **Basis Functions**: Each control point has an associated basis function that determines its influence on the curve at any given point. Key properties:
   - They are non-negative (always $\geq 0$)
   - They sum to 1 at every point (partition of unity)
   - Each basis function has local support (affects only part of the curve)
   - Their shape determines how smoothly the curve transitions

3. **Knot Vector**: A sequence of values that defines where the basis functions start, end, and transition. Multiple knots at a point can create sharp features in the curve.

The B-spline curve is formed by the weighted sum of control points, where the weights are given by the basis functions:

$$
S(t) = \sum_{i=0}^{n} B_i^k(t) C_i
$$

where:
- $C_i$ are the control points
- $B_i^k(t)$ are the basis functions of degree $k$
- $t$ is the parameter along the curve

```python
python draw.py -bspline
```

In KANs, B-splines are used as learnable activation functions on the edges of the network. Each edge has a B-spline defined by coefficients $c_i$ that the network adjusts during training. This allows the activation functions to adapt smoothly to the data, enhancing the network's ability to model complex patterns.

## 4. Mathematical Foundations of KANs

KANs implement the Kolmogorov-Arnold theorem through a structured network architecture that transforms input vectors through layers of learnable univariate functions.

### 4.1 Function Decomposition

Given an input vector $\mathbf{x} = [x_1, x_2, \ldots, x_n]$, KANs approximate a multivariate function $f(\mathbf{x})$ as:

$$
f(\mathbf{x}) = \sum_{q=1}^{2n+1} \Phi_q\left(\sum_{p=1}^n \phi_{q,p}(x_p)\right)
$$

where:
- $\phi_{q,p}(x_p)$ are univariate functions (implemented as B-splines) that transform each input feature
- $\Phi_q$ are combining functions that aggregate the transformed inputs
- $n$ is the input dimension
- The Kolmogorov-Arnold theorem guarantees that $2n+1$ terms are sufficient to approximate any continuous multivariate function.

### 4.2 Layer-wise Transformation

Each layer $l$ in a KAN applies learnable transformations using B-spline functions $\phi_{l,i,j}$:

$$
x_j^{(l+1)} = \sum_{i=1}^{n_l} \phi_{l,i,j}(x_i^{(l)})
$$

where:
- $x_i^{(l)}$ are the outputs from layer $l$
- $x_j^{(l+1)}$ are the inputs to layer $l+1$
- $n_l$ is the number of nodes in layer $l$
- $\phi_{l,i,j}$ are B-spline functions defined by learnable control points

### 4.3 B-spline Implementation

Each univariate function $\phi_{l,i,j}$ is implemented as a B-spline:

$$
\phi_{l,i,j}(t) = \sum_{i=0}^{n} B_i^k(t) c_i
$$

where:
- $B_i^k(t)$ are the basis functions of degree $k$
- $c_i$ are learnable control point coefficients
- $t$ is the input value being transformed

### 4.4 Complete Network Function

The overall KAN output is a composition of layer transformations:

$$
\text{KAN}(\mathbf{x}) = (\Phi_L \circ \Phi_{L-1} \circ \cdots \circ \Phi_0)(\mathbf{x})
$$

where $\Phi_l$ represents the complete transformation at layer $l$.

### 4.5 Key Properties

1. **Universal Approximation**: The network can theoretically approximate any continuous multivariate function.

2. **Interpretability**: Each edge transformation can be visualized and understood through its B-spline representation:

3. **Adaptability**: The control points of each B-spline are learned during training, allowing the network to adapt its activation functions to the specific problem.

4. **Smoothness**: B-spline properties ensure that all transformations are smooth and well-behaved, improving optimization and generalization.

## References

- The Math Behind KAN - [Medium](https://towardsdatascience.com/the-math-behind-kan-kolmogorov-arnold-networks-7c12a164ba95)
- Kolmogorov-Arnold Representation Theorem - [Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem)
