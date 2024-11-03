import numpy as np
from typing import Callable, Tuple

from src.Code.GradientDescent import GradientDescent
from src.Code.PrepareDataset import data_generator
from utils import ReLU


class MultilayerPerceptron:
    def __init__(
            self,
            func: Callable = ReLU,
            number_hidden_layers: int = 1,
            number_units_in_hidden_layers: Tuple[int, int] = (64, 64),
            learning_rate: float = 0.01,
            epochs: int = 100,
    ):
        self.params = None
        self.M = 64
        self.func = func
        self.number_hidden_layers = number_hidden_layers
        self.number_units_in_hidden_layers = number_units_in_hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._initialize_weights()

        self.weights = None
        self.biases = None

    def _initialize_weights(self):
        self.weights = []
        self.biases = []
        input_size = 784  # Assuming input images are flattened 28x28

        for units in self.number_units_in_hidden_layers:
            self.weights.append(np.random.randn(input_size, units) * 0.01)
            self.biases.append(np.zeros((1, units)))
            input_size = units

        # Output layer weights (assuming n_classes is 1 for binary classification)
        self.weights.append(np.random.randn(input_size, 1) * 0.01)
        self.biases.append(np.zeros((1, 1)))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def fit(self, x, y, optimizer):
        N, D = x.shape

        def gradient(x, y, params):
            v, w = params
            z = self.func(np.dot(x, v))  # N x M
            yh = self.softmax(np.dot(z, w))  # N
            dy = yh - y  # N
            dw = np.dot(z.T, dy) / N  # M
            dz = np.dot(dy, w.T) * (z > 0)  # Gradient for ReLU
            dv = np.dot(x.T, dz)
            dparams = [dv, dw]
            return dparams

        w = np.random.randn(self.number_units_in_hidden_layers[-1], 11) * .01
        v = np.random.randn(D, self.number_units_in_hidden_layers[-1]) * .01

        params0 = [v, w]
        self.params = optimizer.run(gradient, x, y, params0)
        return self

    def predict(self, x):
        v, w = self.params
        z = self.func(np.dot(x, v))  # N x M
        yh = self.softmax(np.dot(z, w))  # N
        return yh


if __name__ == '__main__':
    model = MultilayerPerceptron()
    optimizer = GradientDescent()
    train_loader, test_loader = data_generator()

    # for epoch in range(100):

    for x, y in train_loader:
        # Flatten each image in the batch to a 1D vector of 784 elements
        x = np.array(x).reshape(x.shape[0], -1)  # Converts from (32, 28, 28) to (32, 784)
        y = np.eye(11)[y.flatten()]  # One-hot encoding for multi-class labels

        model.fit(x, y, optimizer)
        break
        # print(f"Completed Epoch {epoch + 1}/{100}")

    # Evaluate the model accuracy on the test set
    correct = 0
    total = 0

    for x, y in test_loader:
        # Flatten each image in the batch to a 1D vector of 784 elements
        x = np.array(x).reshape(x.shape[0], -1)  # Converts from (32, 28, 28) to (32, 784)
        y = y.flatten()

        # Make predictions
        y_pred = model.predict(x)
        y_pred = np.argmax(y_pred, axis=1)
        # Calculate accuracy for the batch
        correct += np.sum(y_pred == y)
        total += y.shape[0]
        break
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
