from tensorflow.keras import layers, models
from utils_cnn import *
import numpy as np
from medmnist import OrganAMNIST

model = models.Sequential([
    layers.Input(shape=(128, 128, 1)),  # Input shape for 128x128 grayscale images
    layers.Conv2D(32, (3, 3), activation='relu'),  # Convolutional Layer 1
    layers.MaxPooling2D((2, 2)),  # Pooling Layer 1
    layers.Conv2D(64, (3, 3), activation='relu'),  # Convolutional Layer 2
    layers.MaxPooling2D((2, 2)),  # Pooling Layer 2
    layers.Conv2D(128, (3, 3), activation='relu'),  # Convolutional Layer 3
    layers.MaxPooling2D((2, 2)),  # Pooling Layer 3
    layers.Flatten(),  # Flattening Layer
    layers.Dense(512, activation='relu'),  # Fully Connected Hidden Layer 1
    layers.Dense(256, activation='relu'),  # Fully Connected Hidden Layer 2
    layers.Dense(11, activation='softmax')  # Output Layer for 11 classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_dataset = OrganAMNIST(split="train", download=True, size=128)

mean,std = compute_mean_std(train_dataset)
print(mean,std)


# Load dataset splits
train_dataset = OrganAMNIST(split="train", size=128, transform=lambda img: numpy_transform(img,mean,std))
test_dataset = OrganAMNIST(split="test", size=128, transform=lambda img: numpy_transform(img,mean,std))


train_list, train_label = convert_data_from_loader(train_dataset)
test_list, test_label = convert_data_from_loader(test_dataset)
train_list = train_list.reshape(-1, 128, 128, 1)
train_label = np.array(train_label)  # Convert the list to a numpy array

# Ensure train_label is a 1D array (shape: num_samples,)
train_label = train_label.reshape(-1)

history = model.fit(train_list, train_label, epochs=10)

test_list = test_list.reshape(-1, 128, 128, 1)
test_label = np.array(test_label)  # Convert the list to a numpy array

test_label = test_label.reshape(-1)
loss, accuracy = model.evaluate(test_list, test_label)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")