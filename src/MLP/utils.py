import numpy as np


### Activation Functions ###
def ReLU(x, derivative=False):
    """_summary_
    Computes the ReLU activation function or its derivative.

    Args:
        x (np.ndarray): Input array.
        derivative (bool, optional): If True, computes the derivative. Defaults to False.

    Returns:
        np.ndarray: Output array after applying ReLU or its derivative.
    """
    if derivative:
        return ReLU_derivative(x)
    return np.maximum(0.0, x)


def leaky_ReLU(x, alpha=0.01, derivative=False):
    """_summary_
    Computes the Leaky ReLU activation function or its derivative.

    Args:
        x (np.ndarray): Input array.
        alpha (float, optional): Slope of the negative part of the function. Defaults to 0.01.
        derivative (bool, optional): If True, computes the derivative. Defaults to False.

    Returns:
        np.ndarray: Output array after applying Leaky ReLU or its derivative.
    """
    if derivative:
        return leaky_ReLU_derivative(x, alpha)
    return np.maximum(0.0, x) + alpha * np.minimum(0.0, x)


def sigmoid(x, derivative=False):
    """_summary_
    Computes the sigmoid activation function or its derivative

    Args:
        x (np.ndarray): Input array.
        derivative (bool, optional): If True, computes the derivative. Defaults to False.

    Returns:
        np.ndarray: Output array after applying sigmoid or its derivative.
    """
    if derivative:
        return sigmoid_derivative(x)
    return 1 / (1 + np.exp(-x))


def softmax(x, derivative=False):
    """Computes the softmax activation function or its derivative.

    Args:
        x (np.ndarray): Input array.
        derivative (bool, optional): If True, computes the derivative. Defaults to False.

    Returns:
        np.ndarray: Output array after applying softmax or its derivative.
    """
    if derivative:
        return softmax_derivative(x)
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def tanh(x, derivative=False):
    """_summary_
    Computes the tanh activation function or its derivative.

    Args:
        x (np.ndarray): Input array.
        derivative (bool, optional): If True, computes the derivative. Defaults to False.

    Returns:
        np.ndarray: Output array after applying tanh or its derivative.
    """
    if derivative:
        return tanh_derivative(x)
    return np.tanh(x)

def softmax(x, derivative=False):
    """Computes the softmax activation function or its derivative.

    Args:
        x (np.ndarray): Input array.
        derivative (bool, optional): If True, computes the derivative. Defaults to False.

    Returns:
        np.ndarray: Output array after applying softmax or its derivative.
    """
    if derivative:
        return softmax_derivative(x)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

### Activation Functions Derivatives ###


def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)


def leaky_ReLU_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax_derivative(x):
    return x * (1 - x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def softmax_derivative(x):
    return x * (1 - x)


### Loss Functions ###


def cross_entropy_loss(y_pred, y_true, derivative=False):
    """
    Computes the cross-entropy loss between predicted and true labels or its derivative.

    Args:
        y_pred (np.ndarray): Predicted probabilities.
        y_true (np.ndarray): True labels, one-hot encoded.
        derivative (bool, optional): If True, computes the derivative. Defaults to False.
    Returns:
        float: Cross-entropy loss or its derivative.
    """
    if derivative:
        return cross_entropy_loss_derivative(y_pred, y_true)
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]


def cross_entropy_loss_derivative(y_pred, y_true):
    return y_pred - y_true
