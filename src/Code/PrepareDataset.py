import numpy as np
from medmnist import OrganAMNIST
train_dataset = OrganAMNIST(split="train",download = True)
from MultilayerPerceptron_marc import *
from sklearn.preprocessing import OneHotEncoder

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
    img = (img - mean) / std     # Normalize using calculated mean and std
    img = img.flatten()          # Flatten the image
    return img

# Load dataset splits
train_dataset = OrganAMNIST(split="train", transform=lambda img: numpy_transform(img))
test_dataset = OrganAMNIST(split="test", transform=lambda img: numpy_transform(img))


def convert_data_from_loader (loader):
    data_list = []
    labels_list = []
# Iterate through the DataLoader
    for data, labels in loader:
        data_list.append(data)
        labels_list.append(labels)

    encoder = OneHotEncoder(categories='auto', sparse_output=False, dtype=int)

    one_hot_labels = encoder.fit_transform(labels_list)
    data_list=np.array(data_list)
    return data_list, one_hot_labels


input_size = 28 * 28
mlp = MultilayerPerceptron(input_size=input_size,output_size=11)

train_list,train_label = convert_data_from_loader(train_dataset)
mlp.fit(train_list,train_label)

test_list,test_label = convert_data_from_loader(test_dataset)
y_pred = mlp.predict(test_list)
print(mlp.evaluate_acc(y_pred,test_label))