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

## Neural Network Layers
In neural networks, different types of layers are used depending on the kind of problem we're trying to solve. Two commonly used layer types are **Dense (fully connected)** layers and **Convolutional layers**.

- **Dense Layer**: Dense layers, also known as fully connected layers, are the 'traditional' type of layer that are used in multi-layer perceptron neural networks. In a dense layer, each neuron is connected to every neuron in the previous layer, and each connection has its own weight. This is a totally general purpose connection pattern and makes no assumptions about the features in the input data thus not spatially aware. These layers are often placed towards the end of the network architecture.

- **Convolutional Layer**: Convolutional layers, on the other hand, use a different pattern of connections. Instead of being fully connected to all neurons in the previous layer, neurons in a convolutional layer are only connected to a smaller number of nearby neurons in the previous layer. This local receptive field concept allows the network to focus on low-level features such as edges and textures in the early layers, and then assemble these into higher level features (like shapes or objects) in later layers. Convolutional layers are particularly effective for tasks where spatial relationships matter, such as image and video processing tasks.

### Key Differences
- **Spatial Awareness:** Dense layers are not spatially aware and treat input pixels which are far apart and close together on equal footing. On the other hand, convolutional layers have neurons that only connect to a subset of the input data, and are therefore spatially aware, making them more suitable for tasks such as image recognition where spatial relationships matter.

- **Parameter Efficiency:** Dense layers might not be parameter efficient when dealing with images because they connect each neuron to every neuron in the previous layer, which can be a large number for high-resolution images. Convolutional layers share weights across space in a translationally invariant manner, drastically reducing the number of parameters, and making them more efficient.

- **Usage in Network Architecture:** Dense layers are typically used towards the end of a neural network architecture, often for final classification. Convolutional layers, on the other hand, are usually used in the earlier stages of the network for feature extraction.

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