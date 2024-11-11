import numpy as np
from medmnist import OrganAMNIST
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
train_dataset = OrganAMNIST(split="train")
from MultilayerPerceptron import *
from sklearn.preprocessing import OneHotEncoder

mean_sum = 0.0
squared_sum = 0.0
total_pixels = 0

# Convert each image to a tensor and calculate mean and std
for img, _ in train_dataset:
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.view(-1)

    # Update statistics
    mean_sum += img_tensor.mean()
    squared_sum += img_tensor.pow(2).mean()
    total_pixels += img_tensor.numel()

# Calculate mean and standard deviation
mean = mean_sum / len(train_dataset)
std = torch.sqrt(squared_sum / len(train_dataset) - mean.pow(2))

# Define transformations: normalize and flatten images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,)),
    transforms.Lambda(lambda x: x.view(-1))
])

# Load dataset splits
train_dataset = OrganAMNIST(split="train", transform=transform)
test_dataset = OrganAMNIST(split="test", transform=transform)

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
mlp = MultilayerPerceptron(input_size=input_size, hidden_layers=[64], epochs=100, learning_rate=0.01)


train_list,train_label = convert_data_from_loader(train_loader)
mlp.fit(train_list,train_label)
test_list,test_label = convert_data_from_loader(test_loader)
mlp.evaluate_acc(test_list,test_label)