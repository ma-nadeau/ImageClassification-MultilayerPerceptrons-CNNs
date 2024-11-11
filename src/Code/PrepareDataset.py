from medmnist import OrganAMNIST # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
import torch.utils.data as data # type: ignore
from MultilayerPerceptron import * 
from utils import * 


# Load the OrganAMNIST dataset information and download the train dataset
def load_datasets():
    train_dataset = OrganAMNIST(split="train", download=True)
    test_dataset = OrganAMNIST(split="test", download=True)
    return train_dataset, test_dataset


def preprocess_data(dataset, mean=None, std=None):
    # Normalize and flatten the dataset
    data = dataset.imgs
    data = data.reshape(-1, 28 * 28)

    if mean is None and std is None:
        # Calculate mean and std from the training data only
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

    data = (data - mean) / std

    labels = dataset.labels
    return data, labels,  mean, std


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

    # Train the model
    mlp.fit(X_train, y_train)
    # Predict on the test set
    predictions = mlp.predict(X_test)
    # Calculate the accuracy
    accuracy = mlp.evaluate_acc(y_test, predictions)

    return predictions, accuracy


def main():
    # Load the dataset
    train_dataset, test_dataset = load_datasets()

    # Preprocess training data and calculate mean, std
    train_data, train_labels, mean, std = preprocess_data(train_dataset)

    # Normalize test data using the mean and std from the training data
    test_data, test_labels, _, _ = preprocess_data(test_dataset, mean=mean, std=std)

    # Train and fit
    predictions, accuracy = train_and_fit(train_data, train_labels, test_data, test_labels)
    print("Predictions:", predictions)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
