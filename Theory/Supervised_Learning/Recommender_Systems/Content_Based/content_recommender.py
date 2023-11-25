import tensorflow as tf
from keras.models import Model

user_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32)
])

item_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32)
])

num_user_features = 10
num_item_features = 10

# create the user input and point to the base network
user_input = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(user_input)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
item_input = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(item_input)
vm = tf.linalg.l2_normalize(vm, axis=1)

# measure the similarity between the user and item embeddings
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = Model(inputs=[user_input, item_input], outputs=output)

# specify the cost function and optimization strategy
cost_fn = tf.keras.losses.MeanSquaredError()