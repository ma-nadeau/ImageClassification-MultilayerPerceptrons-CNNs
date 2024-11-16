import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import numpy as np
from medmnist import OrganAMNIST
from utils_cnn import *

# Load the ResNet50 model pre-trained on ImageNet, excluding the top fully connected layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze all convolutional layers to prevent their weights from updating
for layer in base_model.layers:
    layer.trainable = False

# Add new fully connected layers on top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Pool the feature maps to reduce dimensionality
    layers.Dense(256, activation='relu'),  # First fully connected layer
    #layers.Dense(128, activation='relu'),   Second fully connected layer
    layers.Dense(11, activation='softmax')  # Output layer for 11 classes in OrganAMNIST
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_dataset = OrganAMNIST(split="train", download=True, size=128)

mean, std = compute_mean_std(train_dataset)
print(mean,std)

train_dataset = OrganAMNIST(split="train", size=128, transform=lambda img: numpy_transform(img,mean,std))
test_dataset = OrganAMNIST(split="test", size=128, transform=lambda img: numpy_transform(img,mean,std))




train_list, train_label = convert_data_from_loader(train_dataset)
test_list, test_label = convert_data_from_loader(test_dataset)
train_list = train_list.reshape(-1, 128, 128, 1)
train_list = np.repeat(train_list, 3, axis=-1)
train_label = np.array(train_label)  # Convert the list to a numpy array

# Ensure train_label is a 1D array (shape: num_samples,)
train_label = train_label.reshape(-1)

history = model.fit(train_list, train_label, epochs=10)

test_list = test_list.reshape(-1, 128, 128, 1)
test_list = np.repeat(test_list, 3, axis=-1)
test_label = np.array(test_label)  # Convert the list to a numpy array

test_label = test_label.reshape(-1)
loss, accuracy = model.evaluate(test_list, test_label)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")