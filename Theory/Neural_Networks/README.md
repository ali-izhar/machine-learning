# Neural Networks Overview
Neural networks, inspired by the human brain, comprise interconnected layers of neurons. These neurons take inputs, apply functions to them, and pass outputs to subsequent neurons.

## The Neural Network Workflow
Neural networks ingest inputs, process them in hidden layers using adjustable weights, and produce a prediction. Weights are modified during training to recognize patterns, enhancing prediction accuracy. Techniques like `backpropagation` and `gradient descent` adjust these weights to minimize the difference between actual and predicted outputs.

## Key Concepts
The network's `depth` refers to its layer count, excluding the input layer. `Width` denotes the number of neurons in a layer. Input layer neurons equal the dataset's feature count, and output layer neurons match the dataset's class count. The user-specified architecture is the number of hidden layers and their neurons.

## Activation Function
Activation functions introduce non-linearity into neural networks, transforming the weighted sum of inputs and bias. 

> Applied to each hidden layer neuron, and in the output layer for classification models, activation functions have diverse forms, each suited to different tasks.

## Model Training Steps

### 1. Model Architecture
Specify how to compute the output of the model for a given input. Consider the following neural network model with 2 hidden layers (25 and 15 neurons) and 1 output layer (1 neuron):

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(units=25, activation='sigmoid'),
    Dense(units=15, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])
```

### 2. Loss and Cost Functions
Specify a loss function to measure how well the model fits the data. In binary classification, the loss function is called the `logistic loss` which, in statistics, is called the `binary cross-entropy` loss function. The binary cross-entropy loss function is given by:

```python
from tensorflow.keras.losses import BinaryCrossentropy
model.compile(loss=BinaryCrossentropy())
```

For a regression problem, we can specify the `mean squared error` loss function:

```python
from tensorflow.keras.losses import MeanSquaredError
model.compile(loss=MeanSquaredError())
```

### 3. Gradient Descent
Specify an optimization algorithm to minimize the loss function. The most common optimization algorithm is `gradient descent`. In order to compute the gradient of the loss function with respect to the weights and biases of the model, we need to use the `backpropagation` algorithm. The backpropagation algorithm is an efficient way to compute the gradient of the loss function with respect to the weights and biases of a neural network. The backpropagation algorithm is given by:

```python
model.fit(X, y, epochs=10)
```

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