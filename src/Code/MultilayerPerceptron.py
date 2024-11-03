import numpy as np
from typing import Callable, List
from utils import ReLU, cross_entropy_loss_derivative


class MultilayerPerceptron:

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int] = [64, 64],
        number_of_hidden_layers: int = 2,
        activation_function: Callable = ReLU,
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
        bias: bool = True,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.number_of_hidden_layers = number_of_hidden_layers
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.bias = bias
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initialize the weights and biases for the network.
        """
        layer_sizes = (
            [self.input_size]
            + self.hidden_layers[: self.number_of_hidden_layers]
            + [self.output_size]
        )
        self.weights = [
            np.random.randn(layer_sizes[i], layer_sizes[i + 1])  # random initialization
            * np.sqrt(2.0 / layer_sizes[i])  # Scaling the weights
            for i in range(len(layer_sizes) - 1)  # iterate over the layers
        ]

        if self.bias:
            self.biases = [
                np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)
            ]

    def forward(self, X):
        """
        Perform a forward pass through the multilayer perceptron.

        Parameters:
        X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            float: The accuracy of the predictions, calculated as the mean of correct predictions. 
            This value represents the proportion of correctly classified samples out of the total samples.
        tuple: A tuple containing:
            - activations (list of numpy.ndarray): List of activations for each layer, including the input layer.
            - Z_values (list of numpy.ndarray): List of linear combinations (Z) for each layer before applying the activation function.

        """
        activations = [X]
        Z_values = []
        for i in range(len(self.weights)):
            # Calculate the output of the layer
            # Z = A * W
            Z = np.dot(activations[-1], self.weights[i])
            # Add the bias if it is used
            if self.bias:
                # Z = Z + b -> Z = A * W + b
                Z += self.biases[i]
            Z_values.append(Z)

            # Apply the activation function -> A = g(Z)
            A = self.activation_function(Z)
            activations.append(A)

        return activations, Z_values

    def backward(self, X, y, activations, Z_values):
        """
        Perform the backward pass of the neural network, calculating the gradients.

        Parameters:
        X (numpy.ndarray): Input data of shape (m, n), where m is the number of examples and n is the number of features.
        y (numpy.ndarray): True labels of shape (m, k), where m is the number of examples and k is the number of output classes.
        activations (list of numpy.ndarray): List of activations from each layer during the forward pass.
        output (numpy.ndarray): The output of the network from the forward pass.

        Returns:
        tuple: A tuple containing:
            - weight_gradients (list of numpy.ndarray): Gradients of the weights for each layer.
            - bias_gradients (list of numpy.ndarray or None): Gradients of the biases for each layer, or None if biases are not used.
        """
        m, _ = X.shape
        weight_gradients = []
        bias_gradients = []

        # Calculate the gradient of the loss with respect to the output of the network (i.e. dA = dL/dA)
        activation_gradient = cross_entropy_loss_derivative(activations[-1], y)

        # Loop through the layers in reverse order
        for i in reversed(range(len(self.weights))):
            ### Calculate the gradients for the weights and biases ###

            # activation_gradient * derivative of the activation function
            # dL/dZ = dL/dA * g'(A * W + b) -> dZ = dA * g'(A * W + b) -> dZ = dA * g'(Z)
            pre_activation_gradient = activation_gradient * self.activation_function(
                Z_values[i], derivative=True
            )

            # Calculate the gradients for the weights
            # dL/dW = dL/dZ * A.T / m -> dW = A.T * dZ / m
            weight_gradient = np.dot(activations[i].T, pre_activation_gradient) / m

            # Insert the gradients at the beginning of the list
            weight_gradients.insert(0, weight_gradient)

            # Calculate the gradients for the biases
            if self.bias:
                # dL/db = sum(dL/dZ) / m -> db = sum(dZ) / m
                bias_gradient = (
                    np.sum(pre_activation_gradient, axis=0, keepdims=True) / m
                )
                bias_gradients.insert(0, bias_gradient)

            # Calculate the gradients for the next layer
            # dL/dA = dL/dZ * W.T -> dA = dZ * W.T
            activation_gradient = np.dot(pre_activation_gradient, self.weights[i].T)

        return weight_gradients, bias_gradients if self.bias else None

    def update_parameters(self, weight_gradients, bias_gradients):
        """
        Update the weights and biases of the neural network using the provided gradients.

        Parameters:
        weight_gradients (list of numpy.ndarray): Gradients of the weights for each layer.
        bias_gradients (list of numpy.ndarray): Gradients of the biases for each layer.

        Returns:
        None
        """

        # Update the weights and biases using the gradients
        for i in range(len(self.weights)):
            # W = W - learning_rate * dL/dW
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            # Update the biases if they are used
            if self.bias:
                # b = b - learning_rate * dL/db
                self.biases[i] -= self.learning_rate * bias_gradients[i]

    def fit(self, X, y):
        """
        Train the multilayer perceptron on the provided data.

        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).
            y (numpy.ndarray): True labels of shape (n_samples, n_classes).
        """
        # Perform the training loop
        for _ in range(self.epochs):
            # Loop through the dataset in batches
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i : i + self.batch_size]
                y_batch = y[i : i + self.batch_size]
                # Forward pass: compute activations (A) and pre-activations (Z) using Z = A_prev * W + b, A = g(Z)
                activations, Z_values = self.forward(X_batch)
                # Backward pass: compute gradients of the loss with respect to the weights and biases 
                # using dL/dW = dL/dZ * A.T / m and dL/db = sum(dL/dZ) / m
                weight_gradients, bias_gradients = self.backward(
                    X_batch, y_batch, activations, Z_values
                )
                # Update the weights and biases using the gradients
                self.update_parameters(weight_gradients, bias_gradients)

    def predict(self, X):
        """
        Perform a forward pass through the neural network to make predictions.

        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: The output of the network (i.e., the activations of the output layer).
        """
        # Perform a forward pass through the neural network
        activations, _ = self.forward(X)
        # Return the output of the network (i.e. the activations of the output layer)
        return activations[-1]

    def evaluate_acc(self, y, yh):
        """
        Evaluate the accuracy of the model's predictions.

        Args:
            y (numpy.ndarray): True labels of shape (n_samples, n_classes).
            yh (numpy.ndarray): Predicted probabilities of shape (n_samples, n_classes).

        Returns:
            float: The accuracy of the predictions.
        """
        # Convert the predicted probabilities to class labels
        predictions = np.argmax(yh, axis=1)
        # Convert the true labels to class labels
        labels = np.argmax(y, axis=1)
        # Calculate the accuracy as the proportion of correct predictions
        accuracy = np.mean(predictions == labels)
        return accuracy
