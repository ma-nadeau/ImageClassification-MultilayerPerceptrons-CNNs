import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from MultilayerPerceptron import MultilayerPerceptron
from utils import ReLU, leaky_ReLU, tanh, sigmoid
from RegularizationType import Regularization


def create_mlp_with_no_hidden_layer(input_size=28 * 28, output_size=11, epochs=10, batch_size=16, learning_rate=0.001, bias=True):
    """
    Creates a model with no hidden layers, directly mapping the inputs to outputs.

    Returns:
        MultilayerPerceptron: The MLP model with no hidden layers.
    """
    mlp = MultilayerPerceptron(
        input_size=input_size,
        number_of_hidden_layers=0,
        output_size=output_size,
        hidden_layers=[],
        activation_function=ReLU,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        bias=bias,
    )
    return mlp

def create_mlp_with_single_hidden_layer_of_256_units(input_size=28 * 28, output_size=11, epochs=10, batch_size=16, learning_rate=0.001, bias=True):
    """
    Creates a model with a single hidden layer of 256 units.

    Returns:
        MultilayerPerceptron: The MLP model with a single hidden layer.
    """
    mlp = MultilayerPerceptron(
        input_size=input_size,
        number_of_hidden_layers=1,
        output_size=output_size,
        hidden_layers=[256],
        activation_function=ReLU,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        bias=bias,
    )
    return mlp


def create_mlp_with_double_hidden_layer_of_256_units(input_size=28 * 28, output_size=11, epochs=10, batch_size=16, learning_rate=0.001, bias=True):
    """
    Creates a model with two hidden layers of 256 units each.

    Returns:
        MultilayerPerceptron: The MLP model with two hidden layers.
    """
    mlp = MultilayerPerceptron(
        input_size=input_size,
        number_of_hidden_layers=2,
        output_size=output_size,
        hidden_layers=[256, 256],
        activation_function=ReLU,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        bias=bias,
    )
    return mlp


def create_mlp_with_double_hidden_layer_of_256_units_and_leaky_ReLU_activation(input_size=28 * 28, output_size=11, epochs=10, batch_size=16, learning_rate=0.001, bias=True):
    """
    Creates a model with two hidden layers of 256 units each and Leaky ReLU activation function.

    Returns:
        MultilayerPerceptron: The MLP model with two hidden layers and Leaky ReLU activation.
    """
    mlp = MultilayerPerceptron(
        input_size=input_size,
        number_of_hidden_layers=2,
        output_size=output_size,
        hidden_layers=[256, 256],
        activation_function=leaky_ReLU,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        bias=bias,
    )
    return mlp


def create_mlp_with_double_hidden_layer_of_256_and_tanh_activation(input_size=28 * 28, output_size=11, epochs=10, batch_size=16, learning_rate=0.001, bias=True):
    """
    Creates a model with two hidden layers of 256 units each and tanh activation function.

    Returns:
        MultilayerPerceptron: The MLP model with two hidden layers and tanh activation.
    """
    mlp = MultilayerPerceptron(
        input_size=input_size,
        number_of_hidden_layers=2,
        output_size=output_size,
        hidden_layers=[256, 256],
        activation_function=tanh,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        bias=bias,
    )
    return mlp


def create_mlp_with_double_hidden_layer_of_256_units_and_sigmoid_activation(input_size=28 * 28, output_size=11, epochs=10, batch_size=16, learning_rate=0.001, bias=True):
    """
    Creates a model with two hidden layers of 256 units each and sigmoid activation function.

    Returns:
        MultilayerPerceptron: The MLP model with two hidden layers and sigmoid activation.
    """
    mlp = MultilayerPerceptron(
        input_size=input_size,
        number_of_hidden_layers=2,
        output_size=output_size,
        hidden_layers=[256, 256],
        activation_function=sigmoid,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        bias=bias,
    )
    return mlp

def create_mlp_with_double_hidden_layer_of_256_units_and_ReLU_activation_L1(input_size=28 * 28, output_size=11, epochs=10, batch_size=16, learning_rate=0.001, bias=True):
    """
    Creates a model with two hidden layers of 256 units each, Leaky ReLU activation function, and L1 regularization.

    Returns:
        MultilayerPerceptron: The MLP model with two hidden layers, Leaky ReLU activation, and L1 regularization.
    """
    mlp = MultilayerPerceptron(
        input_size=input_size,
        number_of_hidden_layers=2,
        output_size=output_size,
        hidden_layers=[256, 256],
        activation_function=ReLU,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        bias=bias,
        regularization=Regularization.L1,
    )
    return mlp

def create_mlp_with_double_hidden_layer_of_256_units_and_ReLU_activation_L2(input_size=28 * 28, output_size=11, epochs=10, batch_size=16, learning_rate=0.001, bias=True):
    """
    Creates a model with two hidden layers of 256 units each, Leaky ReLU activation function, and L2 regularization.

    Returns:
        MultilayerPerceptron: The MLP model with two hidden layers, Leaky ReLU activation, and L2 regularization.
    """
    mlp = MultilayerPerceptron(
        input_size=input_size,
        number_of_hidden_layers=2,
        output_size=output_size,
        hidden_layers=[256, 256],
        activation_function=ReLU,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        bias=bias,
        regularization=Regularization.L2,
    )
    return mlp
