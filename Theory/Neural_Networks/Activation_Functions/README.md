# Activation Function
Activation functions introduce non-linearity into neural networks, transforming the weighted sum of inputs and bias. 

> Applied to each hidden layer neuron, and in the output layer for classification models, activation functions have diverse forms, each suited to different tasks.

- **Sigmoid:** The sigmoid function, expressed as $σ(x) = 1 / (1 + e^{-x})$, squashes its input into the range (0,1). Its output can be interpreted as a probability, making it ideal for binary classification tasks. However, sigmoid can suffer from the vanishing gradient problem, where the gradients become too small to effectively update the weights during training. The derivative of the sigmoid function is $σ'(x) = σ(x)(1 - σ(x))$.

- **ReLU (Rectified Linear Unit):** The ReLU function, expressed as $f(x) = max(0,x)$, allows positive inputs to pass through unaltered, while zeroing out negative inputs. This encourages sparse activation, speeding up computation and learning. However, neurons can "die" if they output zero, rendering them inactive during training. The derivative of the ReLU function is $f'(x) = 1$ if $x > 0$ and $f'(x) = 0$ if $x < 0$, or undefined if $x = 0$. However, in computer implementations, the derivative is often set to 1 if $x \geq 0$ and $0$ otherwise.

- **Leaky ReLU:** The Leaky ReLU function, expressed as $f(x) = max(0.01x,x)$, is a variant of ReLU that prevents neurons from dying by allowing a small, non-zero gradient when the input is negative. This helps the network continue learning even when a neuron is not active. The derivative of the Leaky ReLU function is $f'(x) = 1$ if $x \geq 0$ and $f'(x) = 0.01$ if $x < 0$.

- **TanH:** The TanH function, expressed as $tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$, is similar to the sigmoid function, but squashes its input into the range (-1,1). It is, mathematically, a scaled or shifted version of the sigmoid function. It is also prone to the vanishing gradient problem. The derivative of the TanH function is $tanh'(x) = 1 - tanh^2(x)$.

- **Linear:** The linear activation function, expressed as $f(x) = x$, allows the input to pass through without transformation. It is primarily used in regression tasks where the output can be any real number. However, it lacks non-linearity, preventing the model from learning complex patterns.

Each activation function transforms the "activation value" as:

$$a_j^{[l]} = g(w_j^{[l]} \cdot a_j^{[l-1]} + b_j^{[l]})$$

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