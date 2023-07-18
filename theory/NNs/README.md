# Neural Networks Overview
Neural networks, inspired by the human brain, comprise interconnected layers of neurons. These neurons take inputs, apply functions to them, and pass outputs to subsequent neurons.

## The Neural Network Workflow
Neural networks ingest inputs, process them in hidden layers using adjustable weights, and produce a prediction. Weights are modified during training to recognize patterns, enhancing prediction accuracy. Techniques like `backpropagation` and `gradient descent` adjust these weights to minimize the difference between actual and predicted outputs.

## Key Concepts
The network's `depth` refers to its layer count, excluding the input layer. `Width` denotes the number of neurons in a layer. Input layer neurons equal the dataset's feature count, and output layer neurons match the dataset's class count. The user-specified architecture is the number of hidden layers and their neurons.

## Activation Function
Activation functions introduce non-linearity into neural networks, transforming the weighted sum of inputs and bias. 

> Applied to each hidden layer neuron, and in the output layer for classification models, activation functions have diverse forms, each suited to different tasks.

- **Sigmoid:** The sigmoid function, expressed as $σ(x) = 1 / (1 + e^{-x})$, squashes its input into the range (0,1). Its output can be interpreted as a probability, making it ideal for binary classification tasks. However, sigmoid can suffer from the vanishing gradient problem, where the gradients become too small to effectively update the weights during training.

- **ReLU (Rectified Linear Unit):** The ReLU function, expressed as $f(x) = max(0,x)$, allows positive inputs to pass through unaltered, while zeroing out negative inputs. This encourages sparse activation, speeding up computation and learning. However, neurons can "die" if they output zero, rendering them inactive during training.

- **Linear:** The linear activation function, expressed as $f(x) = x$, allows the input to pass through without transformation. It is primarily used in regression tasks where the output can be any real number. However, it lacks non-linearity, preventing the model from learning complex patterns.

Each activation function transforms the "activation value" as:

$$a_j^{[l]} = g(w_j^{[l]} \cdot a_j^{[l-1]} + b_j^{[l]})$$

## Choosing the Right Activation Function
Selecting the appropriate activation function depends on the nature of the problem and the type of output you're predicting.

For the **output layer:**

- **Sigmoid** is best for binary classification tasks due to its probability interpretation.
- **Linear activation** is suitable for tasks like predicting stock prices, which can take on any real number (positive or negative).
- **ReLU** is ideal for predicting values that are always non-negative, such as house prices.

For **hidden layers, ReLU** is the most commonly used due to its ability to speed up computation and learning by enabling sparse activation.

## Need for Activation Functions
Activation functions are crucial in neural networks to introduce non-linearity, enabling them to learn from complex data patterns. If every neuron in a neural network were to use a linear activation function, the network would function like linear regression. Regardless of the network's depth, it could only fit linear relationships in data, limiting its utility. 

Let's simplify this concept with a one-hidden-unit network example. If a linear function is used everywhere, the output becomes a linear function of the input, equivalent to using a simple linear regression model.

This limitation arises from the fact that the composition of linear functions is also a linear function. Therefore, a multilayer neural network employing linear activation functions would equate to linear or logistic regression, depending on the output layer function. This would prevent the network from learning complex features and diminish the benefit of multiple layers. Therefore, it's advised not to use linear activation functions in hidden layers. The Rectified Linear Unit (ReLU) is a commonly recommended alternative for hidden layers. Activation functions other than linear ones enable neural networks to tackle a wider range of problems, including binary classification, regression, and multi-category classification.


### Definitions
- **X**: Input data
- **Y**: Output data
- **W**: Weights
- **b**: Bias
- **a**: Activation function
- **L**: Number of layers
- **l**: Layer number
- **n**: Number of neurons in a layer
- **m**: Number of training examples
- **A**: Output of activation function
- **σ**: Sigmoid function
