# from medmnist import OrganAMNIST # type: ignore
# import matplotlib.pyplot as plt # type: ignore
# import numpy as np # type: ignore
# import torch # type: ignore
# import torch.utils.data as data # type: ignore
# from MultilayerPerceptron import *
# from utils import *
#
#
# # Load the OrganAMNIST dataset information and download the train dataset
# def load_datasets():
#     train_dataset = OrganAMNIST(split="train", download=True)
#     test_dataset = OrganAMNIST(split="test", download=True)
#     return train_dataset, test_dataset
#
#
# def preprocess_data(dataset, mean=None, std=None):
#     # Normalize and flatten the dataset
#     data = dataset.imgs
#     data = data.reshape(-1, 28 * 28)
#
#     if mean is None and std is None:
#         # Calculate mean and std from the training data only
#         mean = np.mean(data, axis=0)
#         std = np.std(data, axis=0)
#
#     data = (data - mean) / std
#
#     labels = dataset.labels
#     return data, labels,  mean, std
#
#
# def train_and_fit(X_train, y_train, X_test, y_test):
#     input_size = 28 * 28  # 28x28 images
#     mlp = MultilayerPerceptron(
#         input_size=input_size,
#         number_of_hidden_layers=2,
#         output_size=10,
#         hidden_layers=[64, 64],
#         activation_function=ReLU,
#         epochs=10,
#         batch_size=32,
#         learning_rate=0.01,
#         bias=True,
#     )
#
#     # Train the model
#     mlp.fit(X_train, y_train)
#     # Predict on the test set
#     predictions = mlp.predict(X_test)
#     # Calculate the accuracy
#     accuracy = mlp.evaluate_acc(y_test, predictions)
#
#     return predictions, accuracy
#
#
# def main():
#     # Load the dataset
#     train_dataset, test_dataset = load_datasets()
#
#     # Preprocess training data and calculate mean, std
#     train_data, train_labels, mean, std = preprocess_data(train_dataset)
#
#     # Normalize test data using the mean and std from the training data
#     test_data, test_labels, _, _ = preprocess_data(test_dataset, mean=mean, std=std)
#
#     # Train and fit
#     predictions, accuracy = train_and_fit(train_data, train_labels, test_data, test_labels)
#     print("Predictions:", predictions)
#     print("Accuracy:", accuracy)
#
# if __name__ == "__main__":
#     main()
from medmnist import OrganAMNIST  # type: ignore

train_dataset = OrganAMNIST(split="train", download=True)
from MultilayerPerceptron import *
from sklearn.preprocessing import OneHotEncoder  # type: ignore

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
    variance_sum += ((img - mean) ** 2).sum()  # Sum of squared differences from the mean

# Calculate the standard deviation
std = np.sqrt(variance_sum / total_pixels)

print("Mean across dataset:", mean)
print("Standard deviation across dataset:", std)


# Define transformations: normalize and flatten images
def numpy_transform(img):
    img = (img - mean) / std  # Normalize using calculated mean and std
    img = img.flatten()  # Flatten the image
    return img


# Load dataset splits
train_dataset = OrganAMNIST(split="train", transform=lambda img: numpy_transform(img))
test_dataset = OrganAMNIST(split="test", transform=lambda img: numpy_transform(img))


def convert_data_from_loader(loader):
    data_list = []
    labels_list = []
    # Iterate through the DataLoader
    for data, labels in loader:
        data_list.append(data)
        labels_list.append(labels)

    encoder = OneHotEncoder(categories='auto', sparse_output=False, dtype=int)

    one_hot_labels = encoder.fit_transform(labels_list)
    data_list = np.array(data_list)
    return data_list, one_hot_labels


input_size = 28 * 28
mlp = MultilayerPerceptron(input_size=input_size, output_size=11)

train_list, train_label = convert_data_from_loader(train_dataset)
mlp.fit(train_list, train_label)

test_list, test_label = convert_data_from_loader(test_dataset)
y_pred = mlp.predict(test_list)
print(mlp.evaluate_acc(y_pred, test_label))
