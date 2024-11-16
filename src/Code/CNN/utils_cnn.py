import numpy as np
def compute_mean_std(train_dataset):
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
    return mean ,std

def numpy_transform(img,mean,std):
    img = (img - mean) / std  # Normalize using calculated mean and std
    return img

def convert_data_from_loader(loader):
    data_list = []
    labels_list = []
    # Iterate through the DataLoader
    for data, labels in loader:
        data_list.append(data)
        labels_list.append(labels[0])

    data_list = np.array(data_list)
    return data_list, labels_list