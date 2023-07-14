# Neural Networks
A neural network, also known as an artificial neural network (ANN), is a type of machine learning model that is inspired by the human brain. Neural networks consist of interconnected layers of nodes or "neurons", which are designed to mimic the neurons in a biological brain. Each of these nodes takes in input, applies a function to that input, and then passes the output to nodes in the next layer.

## Intuition behind Neural Networks
In essence, a neural network takes in inputs, which are then processed in hidden layers using weights that are adjusted during training. The model then outputs a prediction as output. The weights are adjusted to find patterns in order to make better predictions. Neural networks learn from the input they process by adjusting the weights in order to predict the correct output. These weights are adjusted via techniques like backpropagation and gradient descent, which aim to minimize the difference between the actual and predicted output.

## Terminology
The number of layers in a neural network is known as the depth of the network. The number of neurons in a layer is known as the width of the network. The number of neurons in the input layer is equal to the number of features in the dataset. The number of neurons in the output layer is equal to the number of classes in the dataset. The number of hidden layers and neurons in the hidden layers is known as the architecture of the neural network and is a hyperparameter that is chosen by the user. When counting the number of layers in a neural network, the input layer is not counted by convention.

## Activation Function
An activation function is a function that is applied to the input of a neuron. It is used to introduce non-linearity into the model. Without an activation function, the neural network would be a linear model. The activation function is applied to the weighted sum of the inputs and the bias. The activation function is applied to the input of each neuron in the hidden layers.

- The activation function is not applied to the input layer. 
- The activation function is applied to the output layer if the model is a classification model. 
- The activation function is not applied to the output layer if the model is a regression model.

The output of the activation function is known as the activation or the activation value. A general formula for the activation value is given below:

$$a_j^l = g(w_j^l \cdot a_j^{l-1} + b_j^l)$$

## Notation
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

## Forward Propagation
Forward propagation is the process of calculating the output of a neural network given the input.

## Backpropagation
Backpropagation is the process of calculating the gradient of the loss function with respect to the weights and biases of the neural network. The gradient is then used to update the weights and biases of the neural network.

## Tensorflow Implementation
Consider the following neural network:

```python
from tensorflow.keras.layers import Dense

layer_1 = Dense(units=3, activation='sigmoid')
layer_2 = Dense(units=1, activation='sigmoid')
model = Sequential([layer_1, layer_2])
```

Alternatively, the neural network can be defined as follows:

```python
model = Sequential([
    Dense(units=3, activation='sigmoid'),
    Dense(units=1, activation='sigmoid')
])
```