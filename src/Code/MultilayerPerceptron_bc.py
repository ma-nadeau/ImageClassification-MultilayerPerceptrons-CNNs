import numpy as np
from typing import Callable, List
from utils import *

# Define activation functions and derivatives

def cross_entropy_loss(y, y_hat):
    m = y.shape[0]
    log_likelihood = -np.log(y_hat[range(m), y.argmax(axis=1)])
    return np.sum(log_likelihood) / m

class MultilayerPerceptron2:
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = [64, 64],
        learning_rate: float = 0.01,
        epochs: int = 100,
        activation_func: Callable = ReLU,
        batch_size: int = 32,
    ):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = 11
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_func = activation_func
        self.batch_size = batch_size
        self.activation_derivative = ReLU_derivative if activation_func == ReLU else leaky_ReLU_derivative  # Only for hidden layers

        # Initialize weights and biases
        self.weights, self.biases = self._initialize_weights()

    def _initialize_weights(self):
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i]) for i in range(len(layer_sizes) - 1)]
        biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]
        return weights, biases

    def forward(self, X):
        activations = [X]
        z_values = []


        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            if i < len(self.weights) - 1:  # Apply activation to hidden layers only
                activations.append(self.activation_func(z))
            else:
                activations.append(softmax(z))  # Softmax for output layer (multi-class)

        return activations, z_values

    def backward(self, activations, z_values, y):
        m = y.shape[0]

        # Output layer error for cross-entropy loss
        delta = activations[-1] - y
        deltas = [delta]
        # Backward pass through hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(deltas[-1], self.weights[i + 1].T) * self.activation_derivative(z_values[i])
            deltas.append(delta)

        deltas.reverse()  # Reverse deltas for weight updates

        # Gradient descent update
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i]) / m
            self.biases[i] -= self.learning_rate * np.mean(deltas[i], axis=0, keepdims=True)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

            for start in range(0, X.shape[0], self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                X_batch, y_batch = X[batch_indices], y[batch_indices]

                activations, z_values = self.forward(X_batch)
                self.backward(activations, z_values, y_batch)

            # Optionally print loss every 10 epochs
            if epoch % 10 == 0:
                loss = cross_entropy_loss(y, self.forward(X)[0][-1])
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)  # Convert softmax output to class prediction

    def evaluate_acc(self, X, y_true):
        y_pred = self.predict(X)
        y_true_classes = np.argmax(y_true, axis=1)
        accuracy = np.mean(y_true_classes == y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy