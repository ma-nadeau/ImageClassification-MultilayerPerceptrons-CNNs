import numpy as np


def ReLU(x):
    return np.maximum(0.0, x)


def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)


def leaky_ReLU(x, alpha=0.01):
    return np.maximum(0.0, x) + alpha * np.minimum(0.0, x)


def leaky_ReLU_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def softmax_derivative(x):
    return x * (1 - x)
