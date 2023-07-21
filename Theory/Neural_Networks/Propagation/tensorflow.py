import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate a binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameters
learning_rate = 0.01
n_epochs = 100
n_features = X.shape[1]
n_hidden = 10
n_classes = 1  # binary classification

# Model weights and biases
W1 = tf.Variable(tf.random.normal([n_features, n_hidden]), name='weight1')
b1 = tf.Variable(tf.zeros([n_hidden]), name='bias1')

W2 = tf.Variable(tf.random.normal([n_hidden, n_classes]), name='weight2')
b2 = tf.Variable(tf.zeros([n_classes]), name='bias2')

# Optimizer
optimizer = tf.optimizers.SGD(learning_rate)

# Training
for epoch in range(n_epochs):
    # Forward pass
    with tf.GradientTape() as tape:
        hidden_layer = tf.nn.relu(tf.matmul(X_train, W1) + b1)
        output_layer = tf.nn.sigmoid(tf.matmul(hidden_layer, W2) + b2)

        # Compute loss
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train, logits=output_layer))

    # Backward pass (compute gradients)
    gradients = tape.gradient(loss, [W1, b1, W2, b2])

    # Update weights and biases
    optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.numpy()}")
