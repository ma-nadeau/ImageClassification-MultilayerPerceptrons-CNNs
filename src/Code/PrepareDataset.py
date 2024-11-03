from medmnist import OrganAMNIST
import ssl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from MultilayerPerceptron import *
from utils import *

# Bypass SSL verification for dataset download
ssl._create_default_https_context = ssl._create_unverified_context


# Load the OrganAMNIST dataset information and download the train dataset
def load_datasets():
    train_dataset = OrganAMNIST(split="train", download=True)
    test_dataset = OrganAMNIST(split="test", download=True)
    return train_dataset, test_dataset


def preprocess_data(dataset):
    # Normalize and flatten the dataset
    data = dataset.imgs
    data = data.reshape(-1, 28 * 28)
    
    # Normalize the data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean) / std
    
    labels = dataset.labels
    return data, labels


def convert_to_tensors(data, labels):
    # Convert to PyTorch tensors
    data_tensor = torch.tensor(data, dtype=torch.float64)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return data_tensor, labels_tensor


def train_and_fit(X_train, y_train, X_test, y_test):
    input_size = 28 * 28  # 28x28 images
    mlp = MultilayerPerceptron(
        input_size=input_size,
        number_of_hidden_layers=2,
        output_size=10,
        hidden_layers=[64, 64],
        activation_function=ReLU,
        epochs=10,
        batch_size=32,
        learning_rate=0.01,
        bias=True,
    )
    # Convert tensors to numpy arrays
    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy()
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy()

    # Train the model
    mlp.fit(X_train_np, y_train_np)
    # Predict on the test set
    predictions = mlp.predict(X_test_np)
    # Calculate the accuracy
    accuracy = mlp.evaluate_acc(y_test_np, predictions)

    return predictions, accuracy


def main():
    # Load the dataset
    train_dataset, test_dataset = load_datasets()

    # Display 10 by 10 images from the dataset
    train_dataset.montage(length=10).show()

    train_data, train_labels = preprocess_data(train_dataset)
    test_data, test_labels = preprocess_data(test_dataset)

    train_data_tensor, train_labels_tensor = convert_to_tensors(
        train_data, train_labels
    )
    test_data_tensor, test_labels_tensor = convert_to_tensors(test_data, test_labels)


    # Select a single image from the training and test datasets
    single_train_data_tensor = train_data_tensor[0].unsqueeze(0)
    single_train_labels_tensor = train_labels_tensor[0].unsqueeze(0)
    single_test_data_tensor = test_data_tensor[0].unsqueeze(0)
    single_test_labels_tensor = test_labels_tensor[0].unsqueeze(0)

    # Train and fit using the single image
    prediction, accuracy = train_and_fit(
        single_train_data_tensor, single_train_labels_tensor, single_test_data_tensor, single_test_labels_tensor
    )
    print(prediction)
    print(accuracy)

if __name__ == "__main__":
    main()
