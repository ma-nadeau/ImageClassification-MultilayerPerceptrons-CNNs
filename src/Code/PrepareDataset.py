import numpy as np
from medmnist import OrganAMNIST
from torch.utils.data import DataLoader
train_dataset = OrganAMNIST(split="train",download = True)
from MultilayerPerceptron_bc import *
from MultilayerPerceptron_marc import *
from sklearn.preprocessing import OneHotEncoder

mean_sum = 0.0
squared_sum = 0.0
total_pixels = 0

# Convert each image to a tensor and calculate mean and std
for img, _ in train_dataset:
    img = np.array(img)
    img = img.reshape(-1) # Flatten the image

    # Update statistics
    mean_sum += img.mean()
    squared_sum += (img ** 2).mean()
    total_pixels += img.shape[0]  # Number of pixels

# Calculate mean and std across all images
mean = mean_sum / total_pixels
std = np.sqrt(squared_sum / total_pixels - mean ** 2)

# Print the results
print("Mean :", mean)
print("Standard deviation :", std)

# Define transformations: normalize and flatten images
def numpy_transform(img):
    img = np.array(img) / 255.0  # Scale to [0, 1] range
    img = (img - mean) / std     # Normalize using calculated mean and std
    img = img.flatten()          # Flatten the image
    return img

# Load dataset splits
train_dataset = OrganAMNIST(split="train", transform=lambda img: numpy_transform(img))
test_dataset = OrganAMNIST(split="test", transform=lambda img: numpy_transform(img))

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


def convert_data_from_loader (loader):
    data_list = []
    labels_list = []
# Iterate through the DataLoader
    for data, labels in loader:
        data_list.append(data.numpy())
        labels_list.append(labels.numpy())
    data_array = np.concatenate(data_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    encoder = OneHotEncoder(categories='auto', sparse_output=False, dtype=int)

    one_hot_labels = encoder.fit_transform(labels_array)
    return data_array, one_hot_labels


input_size = 28 * 28
mlp = MultilayerPerceptron(input_size=input_size,output_size=11)

train_list,train_label = convert_data_from_loader(train_loader)
mlp.fit(train_list,train_label)

test_list,test_label = convert_data_from_loader(test_loader)
y_pred = mlp.predict(test_list)
print(mlp.evaluate_acc(y_pred,test_label))