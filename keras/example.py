#!/usr/bin/python3.13

import tensorflow as tf
import numpy as np
import math
import keras as keras


# output = activation(dot(W, input) + b)
class NativeDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation
        w_shape = (input_size, output_size) # matrix shape
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1) # random value
        self.W = tf.Variable(w_initial_value) # Creates a new variable with value `initial_value`.

        b_shape = (output_size,) # vector shape
        b_initial_value = tf.zeros(b_shape) # zero value vector
        self.b = tf.Variable(b_initial_value) # Create a new variable

    # 前向传播 forward pass
    def __call__(self, inputs):
        # matmul method Multiplies matrix a by matrix b, producing `a * b`.
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        return [self.W, self.b]


class NativeSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels


def one_training_step(model, images_batch, labels_batch):
    # Run forward pass
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions,
        )
        average_loss = tf.reduce_mean(per_sample_losses)
    # 损失值列表 gradients
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss


learning_rate = 1e-3
optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
def update_weights(gradients, weights):
    # for g, w in zip(gradients, weights):
    #     w.assign_sub(g * learning_rate) # assign_sub is `-=` operator of the tensor
    optimizer.apply_gradients(zip(gradients, weights))


def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels, batch_size)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")

def main():
    model = NativeSequential([
        NativeDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
        NativeDense(input_size=512, output_size=10, activation=tf.nn.softmax)
    ])
    assert len(model.weights) == 4

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data(path="mnist.npz")

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255

    fit(model, train_images, train_labels, epochs=20, batch_size=64)

    predictions = model(test_images)
    predictions = predictions.numpy()
    predicted_labels = np.argmax(predictions, axis=1)
    matches = predicted_labels == test_labels
    print(f"accuracy: {matches.mean():.2f}")

if __name__ == '__main__':
    main()
