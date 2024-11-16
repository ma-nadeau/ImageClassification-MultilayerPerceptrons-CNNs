import os

from medmnist import OrganAMNIST  # type: ignore
from MultilayerPerceptron import *
from sklearn.preprocessing import OneHotEncoder  # type: ignore
import numpy as np
from ModelCreation import *


def calculate_mean_and_std(train_dataset):
    # First pass: Calculate the mean
    mean_sum = 0
    total_pixels = 0

    for img, _ in train_dataset:
        img = np.array(img)  # Convert PIL image to NumPy array
        img = img.reshape(-1)  # Flatten the image
        mean_sum += img.sum()  # Total sum of all pixel values
        total_pixels += img.size  # Total number of pixels in the dataset

    # Calculate the mean across all pixels
    mean = mean_sum / total_pixels

    # Second pass: Calculate the variance
    variance_sum = 0

    for img, _ in train_dataset:
        img = np.array(img)
        img = img.reshape(-1)
        variance_sum += (
                (img - mean) ** 2
        ).sum()  # Sum of squared differences from the mean

    # Calculate the standard deviation
    std = np.sqrt(variance_sum / total_pixels)

    return mean, std


def numpy_transform(img, mean, std):
    img = (img - mean) / std  # Normalize using calculated mean and std
    img = img.flatten()  # Flatten the image
    return img


def numpy_flatten(img):
    img = np.array(img)  # Convert PIL image to NumPy array
    img = img.flatten()
    return img


def load_and_normalize_dataset(mean, std, size=28):
    """
    Load and normalize the OrganAMNIST dataset.

    This function loads the training and testing splits of the OrganAMNIST dataset
    and applies normalization using the provided mean and standard deviation.

    Args:
        mean (float): The mean value for normalization.
        std (float): The standard deviation value for normalization.

    Returns:
        tuple: A tuple containing the training and testing datasets.
        :param mean:
        :param std:
        :param size:
    """
    # Load dataset splits
    train_dataset = OrganAMNIST(
        split="train", transform=lambda img: numpy_transform(img, mean, std), size=size,
    )
    test_dataset = OrganAMNIST(
        split="test", transform=lambda img: numpy_transform(img, mean, std), size=size,
    )
    return train_dataset, test_dataset


def load_dataset():
    train_dataset = OrganAMNIST(split="train", transform=lambda img: numpy_flatten(img))
    test_dataset = OrganAMNIST(split="test", transform=lambda img: numpy_flatten(img))
    return train_dataset, test_dataset


def convert_data_from_loader(loader):
    data_list = []
    labels_list = []
    # Iterate through the DataLoader
    for data, labels in loader:
        data_list.append(data)
        labels_list.append(labels)

    encoder = OneHotEncoder(categories="auto", sparse_output=False, dtype=int)

    one_hot_labels = encoder.fit_transform(labels_list)
    data_list = np.array(data_list)
    return data_list, one_hot_labels


def prepare_normalized_dataset(size=28) -> tuple:
    train_dataset = OrganAMNIST(split="train", download=True, size=size)
    mean, std = calculate_mean_and_std(train_dataset)
    train_dataset, test_dataset = load_and_normalize_dataset(mean, std, size)
    train_list, train_label = convert_data_from_loader(train_dataset)
    test_list, test_label = convert_data_from_loader(test_dataset)

    return train_list, train_label, test_list, test_label


def prepare_normalized_dataset_128() -> tuple:
    train_dataset = OrganAMNIST(split="train", download=True)
    mean, std = calculate_mean_and_std(train_dataset)
    train_dataset, test_dataset = load_and_normalize_dataset(mean, std)
    train_list, train_label = convert_data_from_loader(train_dataset)
    test_list, test_label = convert_data_from_loader(test_dataset)

    return train_list, train_label, test_list, test_label


def prepare_unnormalized_dataset() -> tuple:
    train_dataset, test_dataset = load_dataset()
    train_list, train_label = convert_data_from_loader(train_dataset)
    test_list, test_label = convert_data_from_loader(test_dataset)

    return train_list, train_label, test_list, test_label


def compare_basic_mlp_models(train_list, train_label, test_list, test_label):
    """Compare basic MLP models with different hidden layer configurations.

    Args:
        train_list (np.ndarray): Training data.
        train_label (np.ndarray): Training labels.
        test_list (np.ndarray): Testing data.
        test_label (np.ndarray): Testing labels.
    """
    mlp_no_hidden_layer = create_mlp_with_no_hidden_layer()
    mlp_single_hidden_layer = create_mlp_with_single_hidden_layer_of_256_units()
    mlp_double_hidden_layer = create_mlp_with_double_hidden_layer_of_256_units()

    mlp_no_hidden_layer.fit(train_list, train_label)
    mlp_single_hidden_layer.fit(train_list, train_label)
    mlp_double_hidden_layer.fit(train_list, train_label)

    y_pred_no_hidden_layer = mlp_no_hidden_layer.predict(test_list)
    y_pred_single_hidden_layer = mlp_single_hidden_layer.predict(test_list)
    y_pred_double_hidden_layer = mlp_double_hidden_layer.predict(test_list)

    acc_no_hidden_layer = mlp_no_hidden_layer.evaluate_acc(
        y_pred_no_hidden_layer, test_label
    )
    acc_single_hidden_layer = mlp_single_hidden_layer.evaluate_acc(
        y_pred_single_hidden_layer, test_label
    )
    acc_double_hidden_layer = mlp_double_hidden_layer.evaluate_acc(
        y_pred_double_hidden_layer, test_label
    )

    print(f"Accuracy of MLP with no hidden layer: {acc_no_hidden_layer}")
    print(f"Accuracy of MLP with single hidden layer: {acc_single_hidden_layer}")
    print(f"Accuracy of MLP with double hidden layer: {acc_double_hidden_layer}")


def compare_activations_for_256_double_hidden_layers(
        train_list, train_label, test_list, test_label
):
    """Compare MLP models with 2 hidden layers of 256 units each using different activation functions.

    Args:
        train_list (np.ndarray): Training data.
        train_label (np.ndarray): Training labels.
        test_list (np.ndarray): Testing data.
        test_label (np.ndarray): Testing labels.
    """
    mlp_double_hidden_layer_relu = create_mlp_with_double_hidden_layer_of_256_units()
    mlp_double_hidden_layer_leaky_relu = (
        create_mlp_with_double_hidden_layer_of_256_units_and_leaky_ReLU_activation()
    )
    mlp_double_hidden_layer_tanh = (
        create_mlp_with_double_hidden_layer_of_256_and_tanh_activation()
    )

    mlp_double_hidden_layer_relu.fit(train_list, train_label)
    mlp_double_hidden_layer_leaky_relu.fit(train_list, train_label)
    mlp_double_hidden_layer_tanh.fit(train_list, train_label)

    y_pred_double_hidden_layer_relu = mlp_double_hidden_layer_relu.predict(test_list)
    y_pred_double_hidden_layer_leaky_relu = mlp_double_hidden_layer_leaky_relu.predict(
        test_list
    )
    y_pred_double_hidden_layer_tanh = mlp_double_hidden_layer_tanh.predict(test_list)

    acc_double_hidden_layer_relu = mlp_double_hidden_layer_relu.evaluate_acc(
        y_pred_double_hidden_layer_relu, test_label
    )
    acc_double_hidden_layer_leaky_relu = (
        mlp_double_hidden_layer_leaky_relu.evaluate_acc(
            y_pred_double_hidden_layer_leaky_relu, test_label
        )
    )
    acc_double_hidden_layer_tanh = mlp_double_hidden_layer_tanh.evaluate_acc(
        y_pred_double_hidden_layer_tanh, test_label
    )

    print(
        f"Accuracy of MLP with 2 hidden layers of 256 units each using ReLU: {acc_double_hidden_layer_relu}"
    )
    print(
        f"Accuracy of MLP with 2 hidden layers of 256 units each using Leaky ReLU: {acc_double_hidden_layer_leaky_relu}"
    )
    print(
        f"Accuracy of MLP with 2 hidden layers of 256 units each using Tanh: {acc_double_hidden_layer_tanh}"
    )


def compare_L1_and_L2_regularization_for_256_double_hidden_layers_MLP(
        train_list, train_label, test_list, test_label, input_size=28 * 28
):
    """Compare MLP models with 2 hidden layers of 256 units each using L1 and L2 regularization.

    Args:
        train_list (np.ndarray): Training data.
        train_label (np.ndarray): Training labels.
        test_list (np.ndarray): Testing data.
        test_label (np.ndarray): Testing labels.
    """
    mlp_double_hidden_layer_L1 = (
        create_mlp_with_double_hidden_layer_of_256_units_and_ReLU_activation_L1(input_size)
    )
    mlp_double_hidden_layer_L2 = (
        create_mlp_with_double_hidden_layer_of_256_units_and_ReLU_activation_L2(input_size)
    )

    mlp_double_hidden_layer_L1.fit(train_list, train_label)
    mlp_double_hidden_layer_L2.fit(train_list, train_label)

    y_pred_double_hidden_layer_L1 = mlp_double_hidden_layer_L1.predict(test_list)
    y_pred_double_hidden_layer_L2 = mlp_double_hidden_layer_L2.predict(test_list)

    acc_double_hidden_layer_L1 = mlp_double_hidden_layer_L1.evaluate_acc(
        y_pred_double_hidden_layer_L1, test_label
    )
    acc_double_hidden_layer_L2 = mlp_double_hidden_layer_L2.evaluate_acc(
        y_pred_double_hidden_layer_L2, test_label
    )

    print(
        f"Accuracy of MLP with 2 hidden layers of 256 units each using L1 regularization: {acc_double_hidden_layer_L1}"
    )
    print(
        f"Accuracy of MLP with 2 hidden layers of 256 units each using L2 regularization: {acc_double_hidden_layer_L2}"
    )


def evaluate_256_double_hidden_layers_unnormalized_image(
        unnormalized_train_list,
        unnormalized_train_label,
        unnormalized_test_list,
        unnormalized_test_label,
):
    """Evaluate MLP model with 2 hidden layers of 256 units each using unnormalized images.

    Args:
        unnormalized_train_list (np.ndarray): Training data.
        unnormalized_train_label (np.ndarray): Training labels.
        unnormalized_test_list (np.ndarray): Testing data.
        unnormalized_test_label (np.ndarray): Testing labels.
    """
    mlp_double_hidden_layer_unnormalized = (
        create_mlp_with_double_hidden_layer_of_256_units()
    )
    mlp_double_hidden_layer_unnormalized.fit(
        unnormalized_train_list, unnormalized_train_label
    )
    y_pred_double_hidden_layer_unnormalized = (
        mlp_double_hidden_layer_unnormalized.predict(unnormalized_test_list)
    )
    acc_double_hidden_layer_unnormalized = (
        mlp_double_hidden_layer_unnormalized.evaluate_acc(
            y_pred_double_hidden_layer_unnormalized, unnormalized_test_label
        )
    )
    print(
        f"Accuracy of MLP with 2 hidden layers of 256 units each using unnormalized images: {acc_double_hidden_layer_unnormalized}"
    )

def plot_accuracy_for_batch_sizes_learning_rates_and_epochs(
    train_list, train_label, test_list, test_label, learning_rates, batch_sizes, epoch_sizes, output_dir="Results"
):
    """
    Generates three plots:
        1. Train vs. Test Accuracy for different batch sizes (fixed learning rate).
        2. Train vs. Test Accuracy for different learning rates (fixed batch size).
        3. Train vs. Test Accuracy for different epoch sizes (fixed batch size and learning rate).

    Args:
        train_list (np.ndarray): Training data.
        train_label (np.ndarray): Training labels.
        test_list (np.ndarray): Testing data.
        test_label (np.ndarray): Testing labels.
        learning_rates (list): List of learning rates to evaluate.
        batch_sizes (list): List of batch sizes to evaluate.
        epoch_sizes (list): List of epoch sizes to evaluate.
        output_dir (str): Directory where the plots will be saved.
    """
    models = {
        "No Hidden Layer": create_mlp_with_no_hidden_layer,
        "Single Hidden Layer (256 units)": create_mlp_with_single_hidden_layer_of_256_units,
        "Double Hidden Layers (256 units)": create_mlp_with_double_hidden_layer_of_256_units,
        "Double Layers + Leaky ReLU": create_mlp_with_double_hidden_layer_of_256_units_and_leaky_ReLU_activation,
        "Double Layers + Tanh": create_mlp_with_double_hidden_layer_of_256_and_tanh_activation,
        "Double Layers + Sigmoid": create_mlp_with_double_hidden_layer_of_256_units_and_sigmoid_activation,
        "Double Layers + ReLU + L1": create_mlp_with_double_hidden_layer_of_256_units_and_ReLU_activation_L1,
        "Double Layers + ReLU + L2": create_mlp_with_double_hidden_layer_of_256_units_and_ReLU_activation_L2,
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Plot for Batch Sizes
    plt.figure(figsize=(16, 10))
    for model_name, create_model in models.items():
        train_accuracies = []
        test_accuracies = []

        for batch_size in batch_sizes:
            # Use a fixed learning rate (e.g., 0.01)
            model = create_model(learning_rate=0.01, batch_size=batch_size)
            model.fit(train_list, train_label)

            # Evaluate training accuracy
            y_train_pred = model.predict(train_list)
            train_accuracy = model.evaluate_acc(train_label, y_train_pred)
            train_accuracies.append(train_accuracy)

            # Evaluate test accuracy
            y_test_pred = model.predict(test_list)
            test_accuracy = model.evaluate_acc(test_label, y_test_pred)
            test_accuracies.append(test_accuracy)

        # Plot train and test accuracy for this model
        plt.plot(batch_sizes, train_accuracies, label=f'{model_name} - Train')
        plt.plot(batch_sizes, test_accuracies, label=f'{model_name} - Test', linestyle='dashed')

    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")
    plt.title("Train vs. Test Accuracy for Different Batch Sizes (Learning Rate = 0.01)")
    plt.legend()
    plt.grid(True)
    batch_plot_path = os.path.join(output_dir, "train_vs_test_accuracy_batch_sizes.png")
    plt.savefig(batch_plot_path)
    print(f"Plot saved to {batch_plot_path}")
    plt.close()

    # 2. Plot for Learning Rates
    plt.figure(figsize=(16, 10))
    for model_name, create_model in models.items():
        train_accuracies = []
        test_accuracies = []

        for lr in learning_rates:
            # Use a fixed batch size (e.g., 32)
            model = create_model(learning_rate=lr, batch_size=32)
            model.fit(train_list, train_label)

            # Evaluate training accuracy
            y_train_pred = model.predict(train_list)
            train_accuracy = model.evaluate_acc(train_label, y_train_pred)
            train_accuracies.append(train_accuracy)

            # Evaluate test accuracy
            y_test_pred = model.predict(test_list)
            test_accuracy = model.evaluate_acc(test_label, y_test_pred)
            test_accuracies.append(test_accuracy)

        # Plot train and test accuracy for this model
        plt.plot(learning_rates, train_accuracies, label=f'{model_name} - Train')
        plt.plot(learning_rates, test_accuracies, label=f'{model_name} - Test', linestyle='dashed')

    plt.xscale("log")  # Log scale for learning rates
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.title("Train vs. Test Accuracy for Different Learning Rates (Batch Size = 32)")
    plt.legend()
    plt.grid(True)
    learning_rate_plot_path = os.path.join(output_dir, "train_vs_test_accuracy_learning_rates.png")
    plt.savefig(learning_rate_plot_path)
    print(f"Plot saved to {learning_rate_plot_path}")
    plt.close()

    # 3. Plot for Epoch Sizes
    plt.figure(figsize=(16, 10))
    for model_name, create_model in models.items():
        train_accuracies = []
        test_accuracies = []

        for epochs in epoch_sizes:
            # Use a fixed batch size (e.g., 32) and learning rate (e.g., 0.01)
            model = create_model(learning_rate=0.01, batch_size=32, epochs=epochs)
            model.fit(train_list, train_label)

            # Evaluate training accuracy
            y_train_pred = model.predict(train_list)
            train_accuracy = model.evaluate_acc(train_label, y_train_pred)
            train_accuracies.append(train_accuracy)

            # Evaluate test accuracy
            y_test_pred = model.predict(test_list)
            test_accuracy = model.evaluate_acc(test_label, y_test_pred)
            test_accuracies.append(test_accuracy)

        # Plot train and test accuracy for this model
        plt.plot(epoch_sizes, train_accuracies, label=f'{model_name} - Train')
        plt.plot(epoch_sizes, test_accuracies, label=f'{model_name} - Test', linestyle='dashed')

    plt.xlabel("Epoch Size")
    plt.ylabel("Accuracy")
    plt.title("Train vs. Test Accuracy for Different Epoch Sizes (Batch Size = 32, Learning Rate = 0.01)")
    plt.legend()
    plt.grid(True)
    epoch_plot_path = os.path.join(output_dir, "train_vs_test_accuracy_epoch_sizes.png")
    plt.savefig(epoch_plot_path)
    print(f"Plot saved to {epoch_plot_path}")
    plt.close()



if __name__ == "__main__":
    train_list, train_label, test_list, test_label = prepare_normalized_dataset()

    (
        unnormalized_train_list,
        unnormalized_train_label,
        unnormalized_test_list,
        unnormalized_test_label,
    ) = prepare_unnormalized_dataset()

    train_list_128, train_label_128, test_list_128, test_label_128 = (
        prepare_normalized_dataset(size=128)
    )
    #
    # # # Experiment #1
    # compare_basic_mlp_models(train_list, train_label, test_list, test_label)
    #
    # # # Experiment #2
    # compare_activations_for_256_double_hidden_layers(
    #     train_list, train_label, test_list, test_label
    # )
    #
    # # # Experiment #3
    # compare_L1_and_L2_regularization_for_256_double_hidden_layers_MLP(
    #     train_list, train_label, test_list, test_label
    # )
    #
    # # # Experiment #4
    # evaluate_256_double_hidden_layers_unnormalized_image(
    #     unnormalized_train_list,
    #     unnormalized_train_label,
    #     unnormalized_test_list,
    #     unnormalized_test_label,
    # )
    #
    # # Experiment #5 - 128x128 pixels
    # compare_L1_and_L2_regularization_for_256_double_hidden_layers_MLP(
    #     train_list_128, train_label_128, test_list_128, test_label_128, input_size=128 * 128
    # )

    # Experiment #6 - Train vs. Test Accuracy for Learning Rates
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    batch_sizes = [8, 16, 32, 64, 128]
    epoch_sizes = [5, 10, 20, 50, 100]
    plot_accuracy_for_batch_sizes_learning_rates_and_epochs(
        train_list, train_label, test_list, test_label, learning_rates, batch_sizes, epoch_sizes
    )

