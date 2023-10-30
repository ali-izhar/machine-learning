# Activation Function
Activation functions introduce non-linearity into neural networks, transforming the weighted sum of inputs and bias. 

| Activation Function | Equation | Derivative | Keypoints |
| ------------------- | -------- | ---------- | -------- |
| Sigmoid | $σ(x) = 1 / (1 + e^{-x})$ | $σ'(x) = σ(x)(1 - σ(x))$ | Saturated neurons "kill" the gradient; Sigmoid outputs are not zero-centered; Exponentials are computationally expensive |
| ReLU | $f(x) = max(0,x)$ | $f'(x) = 1$ if $x \geq 0$ and $f'(x) = 0$ if $x < 0$ | ReLU neurons can "die" during training; Does not saturate in the + region; ReLU outputs are zero-centered |
| Leaky ReLU | $f(x) = max(0.01x,x)$ | $f'(x) = 1$ if $x \geq 0$ and $f'(x) = 0.01$ if $x < 0$ | Leaky ReLU prevents neurons from dying; Leaky ReLU outputs are zero-centered |
| PReLU | $f(x) = max(\alpha x,x)$ | $f'(x) = 1$ if $x \geq 0$ and $f'(x) = \alpha$ if $x < 0$ | PReLU prevents neurons from dying; PReLU outputs are zero-centered |
| ELU | $f(x) = max(\alpha(e^x - 1),x)$ | $f'(x) = 1$ if $x \geq 0$ and $f'(x) = f(x) + \alpha$ if $x < 0$ | ELU (exponential linear units) prevents neurons from dying; ELU outputs are zero-centered |
| TanH | $tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $tanh'(x) = 1 - tanh^2(x)$ | TanH is a scaled version of the sigmoid function; TanH outputs are zero-centered | Squishes outputs to the range (-1,1) |
| Maxout | $f(x) = max(w_1^T x + b_1, w_2^T x + b_2)$ | $f'(x) = w_1$ if $w_1^T x + b_1 > w_2^T x + b_2$ and $f'(x) = w_2$ if $w_1^T x + b_1 < w_2^T x + b_2$ | Maxout is a generalization of ReLU and Leaky ReLU; Maxout doubles the number of parameters per neuron |


## Choosing the Right Activation Function
Selecting the appropriate activation function depends on the nature of the problem and the type of output you're predicting.

For the **output layer:**

- **Sigmoid** is best for binary classification tasks due to its probability interpretation.
- **Linear activation** is suitable for tasks like predicting stock prices, which can take on any real number (positive or negative).
- **ReLU** is ideal for predicting values that are always non-negative, such as house prices.
- **TanH** is a good choice for multi-class classification tasks.

For **hidden layers, ReLU** is the most commonly used due to its ability to speed up computation and learning by enabling sparse activation.

## Need for Activation Functions
Activation functions are crucial in neural networks to introduce non-linearity, enabling them to learn from complex data patterns. If every neuron in a neural network were to use a linear activation function, the network would function like linear regression. Regardless of the network's depth, it could only fit linear relationships in data, limiting its utility. 

Let's simplify this concept with a one-hidden-unit network example. If a linear function is used everywhere, the output becomes a linear function of the input, equivalent to using a simple linear regression model.

This limitation arises from the fact that the composition of linear functions is also a linear function. Therefore, a multilayer neural network employing linear activation functions would equate to linear or logistic regression, depending on the output layer function. This would prevent the network from learning complex features and diminish the benefit of multiple layers. Therefore, it's advised not to use linear activation functions in hidden layers. The Rectified Linear Unit (ReLU) is a commonly recommended alternative for hidden layers. Activation functions other than linear ones enable neural networks to tackle a wider range of problems, including binary classification, regression, and multi-category classification.