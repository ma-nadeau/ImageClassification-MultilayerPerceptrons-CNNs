import os

from medmnist import OrganAMNIST  # type: ignore
from MultilayerPerceptron import *
from sklearn.preprocessing import OneHotEncoder  # type: ignore
import numpy as np
from ModelCreation import *
import time
import matplotlib.pyplot as plt  # type: ignore


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
        split="train",
        transform=lambda img: numpy_transform(img, mean, std),
        size=size,
    )
    test_dataset = OrganAMNIST(
        split="test",
        transform=lambda img: numpy_transform(img, mean, std),
        size=size,
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

    start_time_no_hidden_layer = time.time()
    mlp_no_hidden_layer.fit(train_list, train_label)
    end_time_no_hidden_layer = time.time()
    diff_no_hidden_layer = end_time_no_hidden_layer - start_time_no_hidden_layer

    start_time_single_hidden_layer = time.time()
    mlp_single_hidden_layer.fit(train_list, train_label)
    end_time_single_hidden_layer = time.time()
    diff_single_hidden_layer = (
        end_time_single_hidden_layer - start_time_single_hidden_layer
    )

    start_time_double_hidden_layer = time.time()
    mlp_double_hidden_layer.fit(train_list, train_label)
    end_time_double_hidden_layer = time.time()
    diff_double_hidden_layer = (
        end_time_double_hidden_layer - start_time_double_hidden_layer
    )

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

    recall_no_hidden_layer = mlp_no_hidden_layer.evaluate_recall(
        y_pred_no_hidden_layer, test_label
    )
    recall_single_hidden_layer = mlp_single_hidden_layer.evaluate_recall(
        y_pred_single_hidden_layer, test_label
    )
    recall_double_hidden_layer = mlp_double_hidden_layer.evaluate_recall(
        y_pred_double_hidden_layer, test_label
    )

    print(f"Accuracy of MLP with no hidden layer: {acc_no_hidden_layer}")
    print(f"Time taken for MLP with no hidden layer: {diff_no_hidden_layer} seconds")
    print(f"Recall of MLP with no hidden layer: {recall_no_hidden_layer}")
    print(f"Accuracy of MLP with single hidden layer: {acc_single_hidden_layer}")
    print(
        f"Time taken for MLP with single hidden layer: {diff_single_hidden_layer} seconds"
    )
    print(f"Recall of MLP with single hidden layer: {recall_single_hidden_layer}")
    print(f"Accuracy of MLP with double hidden layer: {acc_double_hidden_layer}")
    print(
        f"Time taken for MLP with double hidden layer: {diff_double_hidden_layer} seconds"
    )
    print(f"Recall of MLP with double hidden layer: {recall_double_hidden_layer}")

    return (
        acc_no_hidden_layer,
        diff_no_hidden_layer,
        recall_no_hidden_layer,
        acc_single_hidden_layer,
        diff_single_hidden_layer,
        recall_single_hidden_layer,
        acc_double_hidden_layer,
        diff_double_hidden_layer,
        recall_double_hidden_layer,
    )


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

    start_time_double_hidden_layer_relu = time.time()
    mlp_double_hidden_layer_relu.fit(train_list, train_label)
    end_time_double_hidden_layer_relu = time.time()
    diff_double_hidden_layer_relu = (
        end_time_double_hidden_layer_relu - start_time_double_hidden_layer_relu
    )

    start_time_double_hidden_layer_leaky_relu = time.time()
    mlp_double_hidden_layer_leaky_relu.fit(train_list, train_label)
    end_time_double_hidden_layer_leaky_relu = time.time()
    diff_double_hidden_layer_leaky_relu = (
        end_time_double_hidden_layer_leaky_relu
        - start_time_double_hidden_layer_leaky_relu
    )

    start_time_double_hidden_layer_tanh = time.time()
    mlp_double_hidden_layer_tanh.fit(train_list, train_label)
    end_time_double_hidden_layer_tanh = time.time()
    diff_double_hidden_layer_tanh = (
        end_time_double_hidden_layer_tanh - start_time_double_hidden_layer_tanh
    )

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

    recall_double_hidden_layer_relu = mlp_double_hidden_layer_relu.evaluate_recall(
        y_pred_double_hidden_layer_relu, test_label
    )
    recall_double_hidden_layer_leaky_relu = (
        mlp_double_hidden_layer_leaky_relu.evaluate_recall(
            y_pred_double_hidden_layer_leaky_relu, test_label
        )
    )
    recall_double_hidden_layer_tanh = mlp_double_hidden_layer_tanh.evaluate_recall(
        y_pred_double_hidden_layer_tanh, test_label
    )

    print(
        f"Accuracy of MLP with 2 hidden layers of 256 units each using ReLU: {acc_double_hidden_layer_relu}"
    )
    print(
        f"Recall of MLP with 2 hidden layers of 256 units each using ReLU: {recall_double_hidden_layer_relu}"
    )
    print(
        f"Time taken for MLP with 2 hidden layers of 256 units each using ReLU: {diff_double_hidden_layer_relu} seconds"
    )

    print(
        f"Accuracy of MLP with 2 hidden layers of 256 units each using Leaky ReLU: {acc_double_hidden_layer_leaky_relu}"
    )
    print(
        f"Recall of MLP with 2 hidden layers of 256 units each using Leaky ReLU: {recall_double_hidden_layer_leaky_relu}"
    )
    print(
        f"Time taken for MLP with 2 hidden layers of 256 units each using Leaky ReLU: {diff_double_hidden_layer_leaky_relu} seconds"
    )

    print(
        f"Accuracy of MLP with 2 hidden layers of 256 units each using Tanh: {acc_double_hidden_layer_tanh}"
    )
    print(
        f"Recall of MLP with 2 hidden layers of 256 units each using Tanh: {recall_double_hidden_layer_tanh}"
    )
    print(
        f"Time taken for MLP with 2 hidden layers of 256 units each using Tanh: {diff_double_hidden_layer_tanh} seconds"
    )

    return (
        acc_double_hidden_layer_relu,
        diff_double_hidden_layer_relu,
        recall_double_hidden_layer_relu,
        acc_double_hidden_layer_leaky_relu,
        diff_double_hidden_layer_leaky_relu,
        recall_double_hidden_layer_leaky_relu,
        acc_double_hidden_layer_tanh,
        diff_double_hidden_layer_tanh,
        recall_double_hidden_layer_tanh,
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
        create_mlp_with_double_hidden_layer_of_256_units_and_ReLU_activation_L1(
            input_size,
            epochs=2,
            batch_size=16,
            learning_rate=1e-2,
            bias=True,
            regularization_param=1e-1,
        )
    )
    mlp_double_hidden_layer_L2 = (
        create_mlp_with_double_hidden_layer_of_256_units_and_ReLU_activation_L2(
            input_size,
            output_size=11,
            epochs=3,
            batch_size=16,
            learning_rate=1e-3,
            bias=True,
            regularization_param=1e-2,
        )
    )

    start_time_double_hidden_layer_L1 = time.time()
    mlp_double_hidden_layer_L1.fit(train_list, train_label)
    end_time_double_hidden_layer_L1 = time.time()
    diff_time_double_hidden_layer_L1 = (
        end_time_double_hidden_layer_L1 - start_time_double_hidden_layer_L1
    )

    start_time_double_hidden_layer_L2 = time.time()
    mlp_double_hidden_layer_L2.fit(train_list, train_label)
    end_time_double_hidden_layer_L2 = time.time()
    diff_time_double_hidden_layer_L2 = (
        end_time_double_hidden_layer_L2 - start_time_double_hidden_layer_L2
    )

    y_pred_double_hidden_layer_L1 = mlp_double_hidden_layer_L1.predict(test_list)
    y_pred_double_hidden_layer_L2 = mlp_double_hidden_layer_L2.predict(test_list)

    acc_double_hidden_layer_L1 = mlp_double_hidden_layer_L1.evaluate_acc(
        y_pred_double_hidden_layer_L1, test_label
    )
    recall_double_hidden_layer_L1 = mlp_double_hidden_layer_L1.evaluate_recall(
        y_pred_double_hidden_layer_L1, test_label
    )
    acc_double_hidden_layer_L2 = mlp_double_hidden_layer_L2.evaluate_acc(
        y_pred_double_hidden_layer_L2, test_label
    )

    recall_double_hidden_layer_L2 = mlp_double_hidden_layer_L2.evaluate_recall(
        y_pred_double_hidden_layer_L2, test_label
    )

    print(
        f"Accuracy of MLP with 2 hidden layers of 256 units each using L1 regularization: {acc_double_hidden_layer_L1}"
    )
    print(
        f"Recall of MLP with 2 hidden layers of 256 units each using L1 regularization: {recall_double_hidden_layer_L1}"
    )
    print(
        f"Time taken for MLP with 2 hidden layers of 256 units each using L1 regularization: {diff_time_double_hidden_layer_L1} seconds"
    )
    print(
        f"Accuracy of MLP with 2 hidden layers of 256 units each using L2 regularization: {acc_double_hidden_layer_L2}"
    )
    print(
        f"Recall of MLP with 2 hidden layers of 256 units each using L2 regularization: {recall_double_hidden_layer_L2}"
    )
    print(
        f"Time taken for MLP with 2 hidden layers of 256 units each using L2 regularization: {diff_time_double_hidden_layer_L2} seconds"
    )

    return (
        acc_double_hidden_layer_L1,
        diff_time_double_hidden_layer_L1,
        recall_double_hidden_layer_L1,
        acc_double_hidden_layer_L2,
        diff_time_double_hidden_layer_L2,
        recall_double_hidden_layer_L2,
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
    start_time_double_hidden_layer_unnormalized = time.time()
    mlp_double_hidden_layer_unnormalized.fit(
        unnormalized_train_list, unnormalized_train_label
    )
    end_time_double_hidden_layer_unnormalized = time.time()
    diff_time_double_hidden_layer_unnormalized = (
        end_time_double_hidden_layer_unnormalized
        - start_time_double_hidden_layer_unnormalized
    )

    y_pred_double_hidden_layer_unnormalized = (
        mlp_double_hidden_layer_unnormalized.predict(unnormalized_test_list)
    )
    acc_double_hidden_layer_unnormalized = (
        mlp_double_hidden_layer_unnormalized.evaluate_acc(
            y_pred_double_hidden_layer_unnormalized, unnormalized_test_label
        )
    )
    recall_double_hidden_layer_unnormalized = (
        mlp_double_hidden_layer_unnormalized.evaluate_recall(
            y_pred_double_hidden_layer_unnormalized, unnormalized_test_label
        )
    )
    print(
        f"Accuracy of MLP with 2 hidden layers of 256 units each using unnormalized images: {acc_double_hidden_layer_unnormalized}"
    )
    print(
        f"Recall of MLP with 2 hidden layers of 256 units each using unnormalized images: {recall_double_hidden_layer_unnormalized}"
    )
    print(
        f"Time taken for MLP with 2 hidden layers of 256 units each using unnormalized images: {diff_time_double_hidden_layer_unnormalized} seconds"
    )
    return (
        acc_double_hidden_layer_unnormalized,
        diff_time_double_hidden_layer_unnormalized,
        recall_double_hidden_layer_unnormalized,
    )


def plot_accuracy_for_batch_sizes_learning_rates_and_epochs(
    train_list,
    train_label,
    test_list,
    test_label,
    learning_rates,
    batch_sizes,
    epoch_sizes,
    train_list_128,
    train_label_128,
    test_list_128,
    test_label_128,
    output_dir="Results",
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
        "1 Hidden Layer (256 units)": create_mlp_with_single_hidden_layer_of_256_units,
        "2 Hidden Layers (256 units)": create_mlp_with_double_hidden_layer_of_256_units,
        "2 Layers + Leaky ReLU": create_mlp_with_double_hidden_layer_of_256_units_and_leaky_ReLU_activation,
        "2 Layers + Tanh": create_mlp_with_double_hidden_layer_of_256_and_tanh_activation,
        "2 Layers + ReLU + L1 (128*128)": create_mlp_with_double_hidden_layer_of_256_units_and_ReLU_activation_L1,
        "2 Layers + ReLU + L2 (128*128)": create_mlp_with_double_hidden_layer_of_256_units_and_ReLU_activation_L2,
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
            if "128" in model_name:
                model = create_model(
                    learning_rate=0.01,
                    epochs=10,
                    batch_size=batch_size,
                    input_size=128 * 128,
                )
                model.fit(train_list_128, train_label_128)

                # Evaluate training accuracy
                y_train_pred = model.predict(train_list_128)
                train_accuracy = model.evaluate_acc(train_label_128, y_train_pred)
                train_accuracies.append(train_accuracy)

                # Evaluate test accuracy
                y_test_pred = model.predict(test_list_128)
                test_accuracy = model.evaluate_acc(test_label_128, y_test_pred)
                test_accuracies.append(test_accuracy)
            else:
                model = create_model(
                    learning_rate=0.01, epochs=10, batch_size=batch_size
                )
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
        plt.plot(batch_sizes, train_accuracies, label=f"{model_name} - Train")
        plt.plot(
            batch_sizes,
            test_accuracies,
            label=f"{model_name} - Test",
            linestyle="dashed",
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")
    plt.title(
        "Train vs. Test Accuracy for Different Batch Sizes (Learning Rate = 0.01, Epochs = 10)"
    )
    plt.legend()
    plt.grid(True)
    # Save the plot
    result_folder = "../Results-MLP"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Correcting the filename
    result_file = os.path.join(result_folder, "train_vs_test_accuracy_batch_sizes.png")
    plt.savefig(result_file)

    # 2. Plot for Learning Rates
    plt.figure(figsize=(16, 10))
    for model_name, create_model in models.items():
        train_accuracies = []
        test_accuracies = []

        for lr in learning_rates:
            # Use a fixed batch size (e.g., 32)
            if "128" in model_name:
                model = create_model(
                    learning_rate=lr, batch_size=16, epochs=10, input_size=128 * 128
                )

                model.fit(train_list_128, train_label_128)

                # Evaluate training accuracy
                y_train_pred = model.predict(train_list_128)
                train_accuracy = model.evaluate_acc(train_label_128, y_train_pred)
                train_accuracies.append(train_accuracy)

                # Evaluate test accuracy
                y_test_pred = model.predict(test_list_128)
                test_accuracy = model.evaluate_acc(test_label_128, y_test_pred)
                test_accuracies.append(test_accuracy)
            else:
                model = create_model(learning_rate=lr, batch_size=16, epochs=10)

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
        plt.plot(learning_rates, train_accuracies, label=f"{model_name} - Train")
        plt.plot(
            learning_rates,
            test_accuracies,
            label=f"{model_name} - Test",
            linestyle="dashed",
        )

    plt.xscale("log")  # Log scale for learning rates
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.title(
        "Train vs. Test Accuracy for Different Learning Rates (Batch Size = 16, Epochs = 10)"
    )
    plt.legend()
    plt.grid(True)
    # Save the plot
    result_folder = "../Results-MLP"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Correcting the filename
    result_file = os.path.join(
        result_folder, "train_vs_test_accuracy_learning_rates.png"
    )
    plt.savefig(result_file)

    # 3. Plot for Epoch Sizes
    plt.figure(figsize=(16, 10))
    for model_name, create_model in models.items():
        train_accuracies = []
        test_accuracies = []

        for epochs in epoch_sizes:
            if "128" in model_name:
                model = create_model(
                    learning_rate=0.01,
                    batch_size=16,
                    epochs=epochs,
                    input_size=128 * 128,
                )

                model.fit(train_list_128, train_label_128)

                # Evaluate training accuracy
                y_train_pred = model.predict(train_list_128)
                train_accuracy = model.evaluate_acc(train_label_128, y_train_pred)
                train_accuracies.append(train_accuracy)

                # Evaluate test accuracy
                y_test_pred = model.predict(test_list_128)
                test_accuracy = model.evaluate_acc(test_label_128, y_test_pred)
                test_accuracies.append(test_accuracy)
            else:
                model = create_model(learning_rate=0.01, batch_size=16, epochs=epochs)

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
        plt.plot(epoch_sizes, train_accuracies, label=f"{model_name} - Train")
        plt.plot(
            epoch_sizes,
            test_accuracies,
            label=f"{model_name} - Test",
            linestyle="dashed",
        )

    plt.xlabel("Epoch Size")
    plt.ylabel("Accuracy")
    plt.title(
        "Train vs. Test Accuracy for Different Epoch Sizes (Batch Size = 16, Learning Rate = 0.01)"
    )
    plt.legend()
    plt.grid(True)
    # Save the plot
    result_folder = "../Results-MLP"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Correcting the filename
    result_file = os.path.join(result_folder, "train_vs_test_accuracy_epoch_sizes.png")
    plt.savefig(result_file)


def plot_accuracy_for_epochs(
    train_list,
    train_label,
    test_list,
    test_label,
    epoch_sizes,
    train_list_128,
    train_label_128,
    test_list_128,
    test_label_128,
    output_dir="../Results-MLP",
):
    """
    Generates individual plots for each model:
    1. Train vs. Test Accuracy for different epoch sizes.

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
        "1 Hidden Layer (256 units)": create_mlp_with_single_hidden_layer_of_256_units,
        "2 Hidden Layers (256 units)": create_mlp_with_double_hidden_layer_of_256_units,
        "2 Layers + Leaky ReLU": create_mlp_with_double_hidden_layer_of_256_units_and_leaky_ReLU_activation,
        "2 Layers + Tanh": create_mlp_with_double_hidden_layer_of_256_and_tanh_activation,
        "2 Layers + ReLU + L1 (128*128)": create_mlp_with_double_hidden_layer_of_256_units_and_ReLU_activation_L1,
        "2 Layers + ReLU + L2 (128*128)": create_mlp_with_double_hidden_layer_of_256_units_and_ReLU_activation_L2,
        "Double Layers + ReLU + L1": create_mlp_with_double_hidden_layer_of_256_units_and_ReLU_activation_L1_28_by_28,
        "Double Layers + ReLU + L2": create_mlp_with_double_hidden_layer_of_256_units_and_ReLU_activation_L2_28_by_28,
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for model_name, create_model in models.items():
        train_accuracies = []
        test_accuracies = []

        for epochs in epoch_sizes:
            if "128" in model_name:
                model = create_model(
                    learning_rate=0.001,
                    batch_size=16,
                    epochs=epochs,
                    input_size=128 * 128,
                )
                model.fit(train_list_128, train_label_128)

                # Evaluate training accuracy
                y_train_pred = model.predict(train_list_128)
                train_accuracy = model.evaluate_acc(train_label_128, y_train_pred)
                train_accuracies.append(train_accuracy)

                # Evaluate test accuracy
                y_test_pred = model.predict(test_list_128)
                test_accuracy = model.evaluate_acc(test_label_128, y_test_pred)
                test_accuracies.append(test_accuracy)
            else:
                model = create_model(learning_rate=0.01, batch_size=16, epochs=epochs)
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
        plt.figure(figsize=(16, 10))
        plt.plot(
            epoch_sizes, train_accuracies, label=f"{model_name} - Train", marker="o"
        )
        plt.plot(
            epoch_sizes,
            test_accuracies,
            label=f"{model_name} - Test",
            linestyle="dashed",
            marker="o",
        )

        plt.xlabel("Epoch Size")
        plt.ylabel("Accuracy")
        plt.title(
            f"Train vs. Test Accuracy for {model_name} (Batch Size = 16, Learning Rate = 0.01)"
        )
        plt.legend()
        plt.grid(True)
        # Save the plot
        result_file = os.path.join(
            output_dir,
            f"train_vs_test_accuracy_epochs_{model_name.replace(' ', '_')}.png",
        )
        plt.savefig(result_file)
        plt.close()


def regularization_strengths(train_list, train_label, test_list, test_label):
    regularization_strengths = [0.001, 0.01, 0.1, 1.0, 10.0]
    l1_accuracies = {"train": [], "test": []}
    l2_accuracies = {"train": [], "test": []}

    for regularization_strength in regularization_strengths:
        model = create_mlp_with_double_hidden_layer_of_256_units_and_ReLU_activation_L1(
            regularization_param=regularization_strength
        )
        model.fit(train_list, train_label)

        # Evaluate training accuracy
        y_train_pred = model.predict(train_list)
        train_accuracy = model.evaluate_acc(train_label, y_train_pred)
        l1_accuracies["train"].append(train_accuracy)

        # Evaluate test accuracy
        y_test_pred = model.predict(test_list)
        test_accuracy = model.evaluate_acc(test_label, y_test_pred)
        l1_accuracies["test"].append(test_accuracy)

    for regularization_strength in regularization_strengths:
        model = create_mlp_with_double_hidden_layer_of_256_units_and_ReLU_activation_L2(
            regularization_param=regularization_strength
        )
        model.fit(train_list, train_label)

        # Evaluate training accuracy
        y_train_pred = model.predict(train_list)
        train_accuracy = model.evaluate_acc(train_label, y_train_pred)
        l2_accuracies["train"].append(train_accuracy)

        # Evaluate test accuracy
        y_test_pred = model.predict(test_list)
        test_accuracy = model.evaluate_acc(test_label, y_test_pred)
        l2_accuracies["test"].append(test_accuracy)

        # Plot the results
    plt.figure(figsize=(10, 6))

    # Validation accuracy
    plt.plot(
        regularization_strengths, l1_accuracies["train"], label="L1 Train", marker="o"
    )
    plt.plot(
        regularization_strengths, l2_accuracies["train"], label="L2 Train", marker="o"
    )

    # Test accuracy
    plt.plot(
        regularization_strengths,
        l1_accuracies["test"],
        label="L1 Test",
        linestyle="--",
        marker="o",
    )
    plt.plot(
        regularization_strengths,
        l2_accuracies["test"],
        label="L2 Test",
        linestyle="--",
        marker="o",
    )

    plt.xscale("log")
    plt.xlabel("Regularization Strength (log scale)")
    plt.ylabel("Accuracy")
    plt.title("Test and Train Accuracy vs Regularization Strength")
    plt.legend()
    plt.grid(True)

    # Save the plot
    result_folder = "../Results-MLP"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Correcting the filename
    result_file = os.path.join(result_folder, "accuracy_vs_regularization_plot.png")
    plt.savefig(result_file)


    # Prepare data for plotting
    # accuracies = [
    #     acc_no_hidden_layer,
    #     acc_single_hidden_layer,
    #     acc_double_hidden_layer,
    #     acc_double_hidden_layer_relu, 
    #     acc_double_hidden_layer_leaky_relu, 
    #     acc_double_hidden_layer_tanh,
    #     acc_double_hidden_layer_L1, 
    #     acc_double_hidden_layer_L2,
    #     acc_double_hidden_layer_unnormalized,
    #     acc_double_hidden_layer_L1_128,
    #     acc_double_hidden_layer_L2_128
    # ]

    # recalls = [
    #     recall_no_hidden_layer, 
    #     recall_single_hidden_layer, 
    #     recall_double_hidden_layer,
    #     recall_double_hidden_layer_relu, 
    #     recall_double_hidden_layer_leaky_relu, 
    #     recall_double_hidden_layer_tanh,
    #     recall_double_hidden_layer_L1, 
    #     recall_double_hidden_layer_L2,
    #     recall_double_hidden_layer_unnormalized,
    #     recall_double_hidden_layer_L1_128, 
    #     recall_double_hidden_layer_L2_128
    # ]
    # print("\nRecalls: ", recalls)

    # times = [
    #     diff_no_hidden_layer, 
    #     diff_single_hidden_layer,
    #     diff_double_hidden_layer,
    #     diff_double_hidden_layer_relu,
    #     diff_double_hidden_layer_leaky_relu, 
    #     diff_double_hidden_layer_tanh,
    #     diff_time_double_hidden_layer_L1, 
    #     diff_time_double_hidden_layer_L2,
    #     diff_time_double_hidden_layer_unnormalized,
    #     diff_time_double_hidden_layer_L1_128, 
    #     diff_time_double_hidden_layer_L2_128
    # ]
    # print("\nTimes: ", times)

    accuracies = [
        0.5736303296208798,
        0.5757678029024638,
        0.6261109236134549,
        0.6641916976037799,
        0.7037349533130836,
        0.6003892451344358,
        0.59867026662166725,
        0.058274271571605356,
        0.58274271571605356,
        0.6024862189222635,
    ]

    recalls = [
        0.5915649545858628,
        0.5029121421372813,
        0.49398537681332133,
        0.7164147871521428,
        0.7096958813979438,
        0.5558771369181194,
        0.53778006554556067,
        0.0052976610519641235,
        0.47976610519641235,
        0.5311028306470705,
    ]

    times = [
        0.008588075637817383,
        0.05878615379333496,
        38.96800422668457,
        45.72416090965271,
        37.243982553482056,
        25.265174865722656,
        27.940442085266113,
        39.24720788002014,
        246.69194626808167,
        248.51464748382568,
    ]

    labels = [
        "No Hidden Layer (ReLU)",
        "Single Hidden Layer (ReLU)",
        "Double Hidden Layer (ReLU)",
        "Double Hidden Layer (Leaky ReLU)",
        "Double Hidden Layer (Tanh)",
        "Double Hidden Layer (ReLU, L1 reg.)",
        "Double Hidden Layer (ReLU, L2 reg.)",
        "Double Hidden Layer Unnormalized (ReLU)",
        "Double Hidden Layer (128x128, L1 reg. ReLU)",
        "Double Hidden Layer (128x128, L2 reg. ReLU)",
    ]

    # Create scatter plot with color representing time
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        recalls, accuracies, c=times, cmap="viridis", s=100, edgecolors="k"
    )

    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Time (s)")

    # Add labels for each point
    for i in range(len(accuracies)):
        plt.text(recalls[i], accuracies[i], labels[i], fontsize=9, ha="right")

    plt.xlabel("Recall", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Accuracy", fontsize=12, fontweight="bold", labelpad=10)
    plt.title(
        "Accuracy vs Recall for Different MLP Models\n(Color represents Time)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.8)
    plt.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout(pad=2)
    plt.gca().set_aspect("auto")

    # Save the plot
    result_folder = "../Results-MLP"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    result_file = os.path.join(result_folder, "accuracy_vs_recall_scatter_plot2.png")
    plt.savefig(result_file)


def plot_experiment_results():
    # Experiment #1
    (
        acc_no_hidden_layer,
        diff_no_hidden_layer,
        recall_no_hidden_layer,
        acc_single_hidden_layer,
        diff_single_hidden_layer,
        recall_single_hidden_layer,
        acc_double_hidden_layer,
        diff_double_hidden_layer,
        recall_double_hidden_layer,
    ) = compare_basic_mlp_models(train_list, train_label, test_list, test_label)

    # Experiment #2
    (
        acc_double_hidden_layer_relu,
        diff_double_hidden_layer_relu,
        recall_double_hidden_layer_relu,
        acc_double_hidden_layer_leaky_relu,
        diff_double_hidden_layer_leaky_relu,
        recall_double_hidden_layer_leaky_relu,
        acc_double_hidden_layer_tanh,
        diff_double_hidden_layer_tanh,
        recall_double_hidden_layer_tanh,
    ) = compare_activations_for_256_double_hidden_layers(
        train_list, train_label, test_list, test_label
    )

    # Experiment #3
    (
        acc_double_hidden_layer_L1,
        diff_time_double_hidden_layer_L1,
        recall_double_hidden_layer_L1,
        acc_double_hidden_layer_L2,
        diff_time_double_hidden_layer_L2,
        recall_double_hidden_layer_L2,
    ) = compare_L1_and_L2_regularization_for_256_double_hidden_layers_MLP(
        train_list, train_label, test_list, test_label
    )

    # Experiment #4
    (
        acc_double_hidden_layer_unnormalized,
        diff_time_double_hidden_layer_unnormalized,
        recall_double_hidden_layer_unnormalized,
    ) = evaluate_256_double_hidden_layers_unnormalized_image(
        unnormalized_train_list,
        unnormalized_train_label,
        unnormalized_test_list,
        unnormalized_test_label,
    )

    # Experiment #5 - 128x128 pixels
    (
        acc_double_hidden_layer_L1_128,
        diff_time_double_hidden_layer_L1_128,
        recall_double_hidden_layer_L1_128,
        acc_double_hidden_layer_L2_128,
        diff_time_double_hidden_layer_L2_128,
        recall_double_hidden_layer_L2_128,
    ) = compare_L1_and_L2_regularization_for_256_double_hidden_layers_MLP(
        train_list_128,
        train_label_128,
        test_list_128,
        test_label_128,
        input_size=128 * 128,
    )

    # Prepare data for plotting
    accuracies = [
        acc_no_hidden_layer,
        acc_single_hidden_layer,
        acc_double_hidden_layer,
        acc_double_hidden_layer_relu,
        acc_double_hidden_layer_leaky_relu,
        acc_double_hidden_layer_tanh,
        acc_double_hidden_layer_L1,
        acc_double_hidden_layer_L2,
        acc_double_hidden_layer_unnormalized,
        acc_double_hidden_layer_L1_128,
        acc_double_hidden_layer_L2_128,
    ]
    print("\nAccuracies: ", accuracies)
    recalls = [
        recall_no_hidden_layer,
        recall_single_hidden_layer,
        recall_double_hidden_layer,
        recall_double_hidden_layer_relu,
        recall_double_hidden_layer_leaky_relu,
        recall_double_hidden_layer_tanh,
        recall_double_hidden_layer_L1,
        recall_double_hidden_layer_L2,
        recall_double_hidden_layer_unnormalized,
        recall_double_hidden_layer_L1_128,
        recall_double_hidden_layer_L2_128,
    ]
    print("\nRecalls: ", recalls)

    times = [
        diff_no_hidden_layer,
        diff_single_hidden_layer,
        diff_double_hidden_layer,
        diff_double_hidden_layer_relu,
        diff_double_hidden_layer_leaky_relu,
        diff_double_hidden_layer_tanh,
        diff_time_double_hidden_layer_L1,
        diff_time_double_hidden_layer_L2,
        diff_time_double_hidden_layer_unnormalized,
        diff_time_double_hidden_layer_L1_128,
        diff_time_double_hidden_layer_L2_128,
    ]
    print("\nTimes: ", times)

    labels = [
        "No Hidden Layer",
        "Single Hidden Layer",
        "Double Hidden Layer",
        "Double Hidden Layer ReLU",
        "Double Hidden Layer Leaky ReLU",
        "Double Hidden Layer Tanh",
        "Double Hidden Layer L1",
        "Double Hidden Layer L2",
        "Double Hidden Layer Unnormalized",
        "Double Hidden Layer L1 (128x128)",
        "Double Hidden Layer L2 (128x128)",
    ]

    # Create scatter plot with color representing time
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        recalls, accuracies, c=times, cmap="viridis", s=100, edgecolors="k"
    )

    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Time (s)")

    # Add labels for each point
    for i in range(len(accuracies)):
        plt.text(recalls[i], accuracies[i], labels[i], fontsize=9, ha="right")

    plt.xlabel("Recall", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Accuracy", fontsize=12, fontweight="bold", labelpad=10)
    plt.title(
        "Accuracy vs Recall for Different MLP Models\n(Color represents Time)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.8)
    plt.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout(pad=2)
    plt.gca().set_aspect("auto")

    # Save the plot
    result_folder = "../Results-MLP"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    result_file = os.path.join(result_folder, "accuracy_vs_recall_scatter_plot2.png")
    plt.savefig(result_file)


def plot_tanh_and_leaky_relu_with_extra_hidden_layers(
    train_list, train_label, test_list, test_label
):
    hidden_layers = [1, 2, 3, 4, 5, 6, 7]
    tanh_accuracies = {"train": [], "test": []}
    epochs = 10
    leaky_relu_accuracies = {"train": [], "test": []}
    hidden_nodes = [
        [256],
        [256, 256],
        [256, 256, 256],
        [256, 256, 256, 256],
        [256, 256, 256, 256, 256],
        [256, 256, 256, 256, 256, 256],
        [256, 256, 256, 256, 256, 256, 256],
    ]
    for idx, num_hidden_layers in enumerate(hidden_layers):
        # Create and train model with Tanh activation
        model_tanh = create_mlp_with_double_hidden_layer_of_256_and_tanh_activation(
            number_of_hidden_layers=num_hidden_layers,
            hidden_layers=hidden_nodes[idx],
            epochs=epochs,
        )
        model_tanh.fit(train_list, train_label)

        # Evaluate training accuracy
        y_train_pred_tanh = model_tanh.predict(train_list)
        train_accuracy_tanh = model_tanh.evaluate_acc(train_label, y_train_pred_tanh)
        tanh_accuracies["train"].append(train_accuracy_tanh)

        # Evaluate test accuracy
        y_test_pred_tanh = model_tanh.predict(test_list)
        test_accuracy_tanh = model_tanh.evaluate_acc(test_label, y_test_pred_tanh)
        tanh_accuracies["test"].append(test_accuracy_tanh)

        # Create and train model with Leaky ReLU activation
        model_leaky_relu = (
            create_mlp_with_double_hidden_layer_of_256_units_and_leaky_ReLU_activation(
                number_of_hidden_layers=num_hidden_layers,
                hidden_layers=hidden_nodes[idx],
                epochs=epochs,
            )
        )
        model_leaky_relu.fit(train_list, train_label)

        # Evaluate training accuracy
        y_train_pred_leaky_relu = model_leaky_relu.predict(train_list)
        train_accuracy_leaky_relu = model_leaky_relu.evaluate_acc(
            train_label, y_train_pred_leaky_relu
        )
        leaky_relu_accuracies["train"].append(train_accuracy_leaky_relu)

        # Evaluate test accuracy
        y_test_pred_leaky_relu = model_leaky_relu.predict(test_list)
        test_accuracy_leaky_relu = model_leaky_relu.evaluate_acc(
            test_label, y_test_pred_leaky_relu
        )
        leaky_relu_accuracies["test"].append(test_accuracy_leaky_relu)

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Tanh accuracies
    plt.plot(hidden_layers, tanh_accuracies["train"], label="Tanh Train", marker="o")
    plt.plot(
        hidden_layers,
        tanh_accuracies["test"],
        label="Tanh Test",
        linestyle="--",
        marker="o",
    )

    # Leaky ReLU accuracies
    plt.plot(
        hidden_layers,
        leaky_relu_accuracies["train"],
        label="Leaky ReLU Train",
        marker="o",
    )
    plt.plot(
        hidden_layers,
        leaky_relu_accuracies["test"],
        label="Leaky ReLU Test",
        linestyle="--",
        marker="o",
    )

    # Set a larger figure size for better visibility
    plt.xlabel(
        "Number of Hidden Layers", fontsize=12
    )  # Increase font size for readability
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(
        "Accuracy vs Number of Hidden Layers for Tanh and Leaky ReLU",
        fontsize=14,
        fontweight="bold",
    )  # Bold and larger font
    plt.legend(loc="best", fontsize=10)  # Adjust legend placement and font size
    plt.grid(True, linestyle="--", alpha=0.7)  # Add dashed grid lines with transparency
    plt.xticks(fontsize=10)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=10)  # Set font size for y-axis tick labels
    plt.tight_layout()  # Ensure no elements are cut off

    # Save the plot
    result_folder = "../Results-MLP"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    result_file = os.path.join(
        result_folder, "accuracy_vs_hidden_layers_tanh_leaky_relu.png"
    )
    plt.savefig(result_file)


def plot_recall_tanh_and_leaky_relu_with_extra_hidden_layers(
    train_list, train_label, test_list, test_label
):
    hidden_layers = [1, 2, 3, 4, 5, 6, 7]
    epochs = 10
    tanh_recalls = {"train": [], "test": []}
    leaky_relu_recalls = {"train": [], "test": []}
    hidden_nodes = [
        [256],
        [256, 256],
        [256, 256, 256],
        [256, 256, 256, 256],
        [256, 256, 256, 256, 256],
        [256, 256, 256, 256, 256, 256],
        [256, 256, 256, 256, 256, 256, 256],
    ]

    for idx, num_hidden_layers in enumerate(hidden_layers):
        # Create and train model with Tanh activation
        model_tanh = create_mlp_with_double_hidden_layer_of_256_and_tanh_activation(
            number_of_hidden_layers=num_hidden_layers,
            hidden_layers=hidden_nodes[idx],
            epochs=epochs,
        )
        model_tanh.fit(train_list, train_label)

        # Evaluate training recall
        y_train_pred_tanh = model_tanh.predict(train_list)
        train_recall_tanh = model_tanh.evaluate_recall(train_label, y_train_pred_tanh)
        tanh_recalls["train"].append(train_recall_tanh)

        # Evaluate test recall
        y_test_pred_tanh = model_tanh.predict(test_list)
        test_recall_tanh = model_tanh.evaluate_recall(test_label, y_test_pred_tanh)
        tanh_recalls["test"].append(test_recall_tanh)

        # Create and train model with Leaky ReLU activation
        model_leaky_relu = (
            create_mlp_with_double_hidden_layer_of_256_units_and_leaky_ReLU_activation(
                number_of_hidden_layers=num_hidden_layers,
                hidden_layers=hidden_nodes[idx],
                epochs=epochs,
            )
        )
        model_leaky_relu.fit(train_list, train_label)

        # Evaluate training recall
        y_train_pred_leaky_relu = model_leaky_relu.predict(train_list)
        train_recall_leaky_relu = model_leaky_relu.evaluate_recall(
            train_label, y_train_pred_leaky_relu
        )
        leaky_relu_recalls["train"].append(train_recall_leaky_relu)

        # Evaluate test recall
        y_test_pred_leaky_relu = model_leaky_relu.predict(test_list)
        test_recall_leaky_relu = model_leaky_relu.evaluate_recall(
            test_label, y_test_pred_leaky_relu
        )
        leaky_relu_recalls["test"].append(test_recall_leaky_relu)

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Tanh recalls
    plt.plot(hidden_layers, tanh_recalls["train"], label="Tanh Train", marker="o")
    plt.plot(
        hidden_layers,
        tanh_recalls["test"],
        label="Tanh Test",
        linestyle="--",
        marker="o",
    )

    # Leaky ReLU recalls
    plt.plot(
        hidden_layers, leaky_relu_recalls["train"], label="Leaky ReLU Train", marker="o"
    )
    plt.plot(
        hidden_layers,
        leaky_relu_recalls["test"],
        label="Leaky ReLU Test",
        linestyle="--",
        marker="o",
    )

    # Set a larger figure size for better visibility
    plt.xlabel(
        "Number of Hidden Layers", fontsize=12
    )  # Increase font size for readability
    plt.ylabel("Recall", fontsize=12)
    plt.title(
        "Recall vs Number of Hidden Layers for Tanh and Leaky ReLU",
        fontsize=14,
        fontweight="bold",
    )  # Bold and larger font
    plt.legend(loc="best", fontsize=10)  # Adjust legend placement and font size
    plt.grid(True, linestyle="--", alpha=0.7)  # Add dashed grid lines with transparency
    plt.xticks(fontsize=10)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=10)  # Set font size for y-axis tick labels
    plt.tight_layout()  # Ensure no elements are cut off

    # Save the plot
    result_folder = "../Results-MLP"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    result_file = os.path.join(
        result_folder, "recall_vs_hidden_layers_tanh_leaky_relu.png"
    )
    plt.savefig(result_file)


def plot_tanh_and_leaky_relu_with_increasing_width(
    train_list, train_label, test_list, test_label
):
    hidden_units = [64, 128, 256, 512, 1024]
    tanh_accuracies = {"train": [], "test": []}
    epochs = 10
    leaky_relu_accuracies = {"train": [], "test": []}

    for num_hidden_units in hidden_units:
        # Create and train model with Tanh activation
        model_tanh = create_mlp_with_double_hidden_layer_of_256_and_tanh_activation(
            hidden_layers=[num_hidden_units, num_hidden_units], epochs=epochs
        )
        model_tanh.fit(train_list, train_label)

        # Evaluate training accuracy
        y_train_pred_tanh = model_tanh.predict(train_list)
        train_accuracy_tanh = model_tanh.evaluate_acc(train_label, y_train_pred_tanh)
        tanh_accuracies["train"].append(train_accuracy_tanh)

        # Evaluate test accuracy
        y_test_pred_tanh = model_tanh.predict(test_list)
        test_accuracy_tanh = model_tanh.evaluate_acc(test_label, y_test_pred_tanh)
        tanh_accuracies["test"].append(test_accuracy_tanh)

        # Create and train model with Leaky ReLU activation
        model_leaky_relu = (
            create_mlp_with_double_hidden_layer_of_256_units_and_leaky_ReLU_activation(
                hidden_layers=[num_hidden_units, num_hidden_units], epochs=epochs
            )
        )
        model_leaky_relu.fit(train_list, train_label)

        # Evaluate training accuracy
        y_train_pred_leaky_relu = model_leaky_relu.predict(train_list)
        train_accuracy_leaky_relu = model_leaky_relu.evaluate_acc(
            train_label, y_train_pred_leaky_relu
        )
        leaky_relu_accuracies["train"].append(train_accuracy_leaky_relu)

        # Evaluate test accuracy
        y_test_pred_leaky_relu = model_leaky_relu.predict(test_list)
        test_accuracy_leaky_relu = model_leaky_relu.evaluate_acc(
            test_label, y_test_pred_leaky_relu
        )
        leaky_relu_accuracies["test"].append(test_accuracy_leaky_relu)

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Tanh accuracies
    plt.plot(hidden_units, tanh_accuracies["train"], label="Tanh Train", marker="o")
    plt.plot(
        hidden_units,
        tanh_accuracies["test"],
        label="Tanh Test",
        linestyle="--",
        marker="o",
    )

    # Leaky ReLU accuracies
    plt.plot(
        hidden_units,
        leaky_relu_accuracies["train"],
        label="Leaky ReLU Train",
        marker="o",
    )
    plt.plot(
        hidden_units,
        leaky_relu_accuracies["test"],
        label="Leaky ReLU Test",
        linestyle="--",
        marker="o",
    )

    # Set a larger figure size for better visibility
    plt.xlabel(
        "Number of Hidden Units", fontsize=12
    )  # Increase font size for readability
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(
        "Accuracy vs Number of Hidden Units for Tanh and Leaky ReLU",
        fontsize=14,
        fontweight="bold",
    )  # Bold and larger font
    plt.legend(loc="best", fontsize=10)  # Adjust legend placement and font size
    plt.grid(True, linestyle="--", alpha=0.7)  # Add dashed grid lines with transparency
    plt.xticks(fontsize=10)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=10)  # Set font size for y-axis tick labels
    plt.tight_layout()  # Ensure no elements are cut off

    # Save the plot
    result_folder = "../Results-MLP"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    result_file = os.path.join(
        result_folder, "accuracy_vs_hidden_units_tanh_leaky_relu.png"
    )
    plt.savefig(result_file)


def plot_recall_tanh_and_leaky_relu_with_increasing_width(
    train_list, train_label, test_list, test_label
):
    hidden_units = [64, 128, 256, 512, 1024]
    epochs = 10
    tanh_recalls = {"train": [], "test": []}
    leaky_relu_recalls = {"train": [], "test": []}

    for num_hidden_units in hidden_units:
        # Create and train model with Tanh activation
        model_tanh = create_mlp_with_double_hidden_layer_of_256_and_tanh_activation(
            hidden_layers=[num_hidden_units, num_hidden_units], epochs=epochs
        )
        model_tanh.fit(train_list, train_label)

        # Evaluate training recall
        y_train_pred_tanh = model_tanh.predict(train_list)
        train_recall_tanh = model_tanh.evaluate_recall(train_label, y_train_pred_tanh)
        tanh_recalls["train"].append(train_recall_tanh)

        # Evaluate test recall
        y_test_pred_tanh = model_tanh.predict(test_list)
        test_recall_tanh = model_tanh.evaluate_recall(test_label, y_test_pred_tanh)
        tanh_recalls["test"].append(test_recall_tanh)

        # Create and train model with Leaky ReLU activation
        model_leaky_relu = (
            create_mlp_with_double_hidden_layer_of_256_units_and_leaky_ReLU_activation(
                hidden_layers=[num_hidden_units, num_hidden_units], epochs=epochs
            )
        )
        model_leaky_relu.fit(train_list, train_label)

        # Evaluate training recall
        y_train_pred_leaky_relu = model_leaky_relu.predict(train_list)
        train_recall_leaky_relu = model_leaky_relu.evaluate_recall(
            train_label, y_train_pred_leaky_relu
        )
        leaky_relu_recalls["train"].append(train_recall_leaky_relu)

        # Evaluate test recall
        y_test_pred_leaky_relu = model_leaky_relu.predict(test_list)
        test_recall_leaky_relu = model_leaky_relu.evaluate_recall(
            test_label, y_test_pred_leaky_relu
        )
        leaky_relu_recalls["test"].append(test_recall_leaky_relu)

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Tanh recalls
    plt.plot(hidden_units, tanh_recalls["train"], label="Tanh Train", marker="o")
    plt.plot(
        hidden_units,
        tanh_recalls["test"],
        label="Tanh Test",
        linestyle="--",
        marker="o",
    )

    # Leaky ReLU recalls
    plt.plot(
        hidden_units, leaky_relu_recalls["train"], label="Leaky ReLU Train", marker="o"
    )
    plt.plot(
        hidden_units,
        leaky_relu_recalls["test"],
        label="Leaky ReLU Test",
        linestyle="--",
        marker="o",
    )

    # Set a larger figure size for better visibility
    plt.xlabel(
        "Number of Hidden Units", fontsize=12
    )  # Increase font size for readability
    plt.ylabel("Recall", fontsize=12)
    plt.title(
        "Recall vs Number of Hidden Units for Tanh and Leaky ReLU",
        fontsize=14,
        fontweight="bold",
    )  # Bold and larger font
    plt.legend(loc="best", fontsize=10)  # Adjust legend placement and font size
    plt.grid(True, linestyle="--", alpha=0.7)  # Add dashed grid lines with transparency
    plt.xticks(fontsize=10)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=10)  # Set font size for y-axis tick labels
    plt.tight_layout()  # Ensure no elements are cut off

    # Save the plot
    result_folder = "../Results-MLP"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    result_file = os.path.join(
        result_folder, "recall_vs_hidden_units_tanh_leaky_relu.png"
    )
    plt.savefig(result_file)


def compare_sigmoid_and_softmax(train_list, train_label, test_list, test_label):
    mlp_sigmoid = (
        create_mlp_with_double_hidden_layer_of_256_units_and_sigmoid_activation()
    )
    mlp_softmax = (
        create_mlp_with_double_hidden_layer_of_256_units_and_softmax_activation()
    )

    start_time_sigmoid = time.time()
    mlp_sigmoid.fit(train_list, train_label)
    end_time_sigmoid = time.time()
    diff_sigmoid = end_time_sigmoid - start_time_sigmoid

    start_time_softmax = time.time()
    mlp_softmax.fit(train_list, train_label)
    end_time_softmax = time.time()
    diff_softmax = end_time_softmax - start_time_softmax

    y_pred_sigmoid = mlp_sigmoid.predict(test_list)
    y_pred_softmax = mlp_softmax.predict(test_list)

    acc_sigmoid = mlp_sigmoid.evaluate_acc(y_pred_sigmoid, test_label)
    acc_softmax = mlp_softmax.evaluate_acc(y_pred_softmax, test_label)

    recall_sigmoid = mlp_sigmoid.evaluate_recall(y_pred_sigmoid, test_label)
    recall_softmax = mlp_softmax.evaluate_recall(y_pred_softmax, test_label)

    print(f"Accuracy of MLP with sigmoid activation: {acc_sigmoid}")
    print(f"Time taken for MLP with sigmoid activation: {diff_sigmoid} seconds")
    print(f"Recall of MLP with sigmoid activation: {recall_sigmoid}")
    print(f"Accuracy of MLP with softmax activation: {acc_softmax}")
    print(f"Time taken for MLP with softmax activation: {diff_softmax} seconds")
    print(f"Recall of MLP with softmax activation: {recall_softmax}")

    return (
        acc_sigmoid,
        diff_sigmoid,
        recall_sigmoid,
        acc_softmax,
        diff_softmax,
        recall_softmax,
    )


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

    Experiment #1
    (
        acc_no_hidden_layer,
        diff_no_hidden_layer,
        recall_no_hidden_layer,
        acc_single_hidden_layer,
        diff_single_hidden_layer,
        recall_single_hidden_layer,
        acc_double_hidden_layer,
        diff_double_hidden_layer,
        recall_double_hidden_layer,
    ) = compare_basic_mlp_models(train_list, train_label, test_list, test_label)

    # Experiment #2
    (
        acc_double_hidden_layer_relu,
        diff_double_hidden_layer_relu,
        recall_double_hidden_layer_relu,
        acc_double_hidden_layer_leaky_relu,
        diff_double_hidden_layer_leaky_relu,
        recall_double_hidden_layer_leaky_relu,
        acc_double_hidden_layer_tanh,
        diff_double_hidden_layer_tanh,
        recall_double_hidden_layer_tanh,
    ) = compare_activations_for_256_double_hidden_layers(
        train_list, train_label, test_list, test_label
    )

    # Experiment #3
    (
        acc_double_hidden_layer_L1,
        diff_time_double_hidden_layer_L1,
        recall_double_hidden_layer_L1,
        acc_double_hidden_layer_L2,
        diff_time_double_hidden_layer_L2,
        recall_double_hidden_layer_L2,
    ) = compare_L1_and_L2_regularization_for_256_double_hidden_layers_MLP(
        train_list, train_label, test_list, test_label
    )

    # Experiment #4
    acc_double_hidden_layer_unnormalized, diff_time_double_hidden_layer_unnormalized, recall_double_hidden_layer_unnormalized = evaluate_256_double_hidden_layers_unnormalized_image(
        unnormalized_train_list,
        unnormalized_train_label,
        unnormalized_test_list,
        unnormalized_test_label,
    )

    # Experiment #5 - 128x128 pixels
    (
        acc_double_hidden_layer_L1_128,
        diff_time_double_hidden_layer_L1_128,
        recall_double_hidden_layer_L1_128,
        acc_double_hidden_layer_L2_128,
        diff_time_double_hidden_layer_L2_128,
        recall_double_hidden_layer_L2_128,
    ) = compare_L1_and_L2_regularization_for_256_double_hidden_layers_MLP(
        train_list_128,
        train_label_128,
        test_list_128,
        test_label_128,
        input_size=128 * 128,
    )

    # Experiment #6 - Train vs. Test Accuracy for Learning Rates
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    batch_sizes = [8, 16, 32, 64, 128]
    epoch_sizes = [5, 10, 20, 50, 100]
    plot_accuracy_for_batch_sizes_learning_rates_and_epochs(
        train_list, train_label, test_list, test_label, learning_rates, batch_sizes, epoch_sizes,
        train_list_128, train_label_128, test_list_128, test_label_128
    )

    epoch_sizes = [1, 2, 5, 10, 15]
    plot_accuracy_for_epochs(
        train_list,
        train_label,
        test_list,
        test_label,
        epoch_sizes,
        train_list_128,
        train_label_128,
        test_list_128,
        test_label_128,
    )
    regularization_strengths(train_list, train_label, test_list, test_label)

    # Call the function to plot Tanh and Leaky ReLU with increasing hidden layers and width
    plot_tanh_and_leaky_relu_with_extra_hidden_layers(train_list, train_label, test_list, test_label)
    plot_recall_tanh_and_leaky_relu_with_extra_hidden_layers(train_list, train_label, test_list, test_label)
    plot_tanh_and_leaky_relu_with_increasing_width(train_list, train_label, test_list, test_label)
    plot_recall_tanh_and_leaky_relu_with_increasing_width(train_list, train_label, test_list, test_label)


    # Sigmoid and Softmax
    compare_sigmoid_and_softmax(train_list, train_label, test_list, test_label)
