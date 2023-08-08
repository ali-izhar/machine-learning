import tensorflow as tf


class CollaborativeFiltering:
    def __init__(self, num_users, num_movies, num_features, lambda_=0.0):
        self.num_users = num_users
        self.num_movies = num_movies
        self.num_features = num_features
        self.lambda_ = lambda_

    def fit(self, X, Y, R, iterations=200, learning_rate=0.001):
        self.X = tf.Variable(X, dtype=tf.float32)
        self.W = tf.Variable(Y, dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([1]), dtype=tf.float32)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        for iter in range(iterations):
            with tf.GradientTape() as tape:
                cost_value = cofiCostFuncV(
                    self.X, self.W, self.b, Y, R, self.num_users, self.num_movies, self.lambda_
                )

            grads = tape.gradient(cost_value, [self.X, self.W, self.b])

            optimizer.apply_gradients(zip(grads, [self.X, self.W, self.b]))

            if (iter + 1) % 20 == 0:
                print("Iteration: %d, cost: %f" % (iter + 1, cost_value))

    def predict(self, X, R):
        return tf.matmul(self.X, tf.transpose(self.W)) + self.b
    
