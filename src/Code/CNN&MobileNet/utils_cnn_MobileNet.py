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
    variance_sum = 0import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from medmnist import OrganAMNIST
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time


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



def create_cnn_model(input_shape, num_classes, num_filters=(4, 8, 12), kernel_size=(3, 3), stride=(1, 1), padding='valid', is_large=False):
    """
    Creates a CNN model with configurable convolutional layer hyperparameters.

    Args:
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of output classes.
        num_filters (tuple): A tuple specifying the number of filters for each convolutional layer.
        kernel_size (tuple): The size of the kernel for convolutional layers.
        stride (tuple): The stride of the convolutional layers.
        padding (str): Padding method, either 'valid' or 'same'.
        is_large (bool): Flag to create a larger model with additional layers for larger input images.

    Returns:
        keras.models.Sequential: Compiled CNN model.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # First convolutional block
    model.add(layers.Conv2D(num_filters[0], kernel_size, strides=stride, padding=padding, activation='relu'))
    if is_large:
        model.add(layers.MaxPooling2D((2, 2)))

    # Second convolutional block
    model.add(layers.Conv2D(num_filters[1], kernel_size, strides=stride, padding=padding, activation='relu'))
    if is_large:
        model.add(layers.MaxPooling2D((2, 2)))

    # Optional larger model with more layers
    if is_large:
        model.add(layers.Conv2D(num_filters[2], kernel_size, strides=stride, padding=padding, activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    if is_large:
        model.add(layers.Dense(512, activation='relu'))

    # Flatten the output and add fully connected layers
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def preprocess_data_cnn(size, mean=None, std=None, validation_split=0.2,validation = True):
    """
    Prepares and preprocesses the training, validation, and test data.

    Args:
        size (int): Image size (e.g., 28 or 128).
        mean (float): Mean value for normalization.
        std (float): Standard deviation value for normalization.
        validation_split (float): Proportion of training data used for validation.

    Returns:
        tuple: Preprocessed datasets (train, validation, and test sets).
    """
    train_dataset = OrganAMNIST(split="train", size=size, transform=lambda img: numpy_transform(img, mean, std))
    test_dataset = OrganAMNIST(split="test", size=size, transform=lambda img: numpy_transform(img, mean, std))

    train_list, train_label = convert_data_from_loader(train_dataset)
    test_list, test_label = convert_data_from_loader(test_dataset)

    train_list = train_list.reshape(-1, size, size, 1)
    test_list = test_list.reshape(-1, size, size, 1)
    train_label = np.array(train_label).reshape(-1)
    test_label = np.array(test_label).reshape(-1)

    # Split train data into training and validation sets
    if validation:
        train_list, val_list, train_label, val_label = train_test_split(
            train_list, train_label, test_size=validation_split, random_state=42
        )

        return train_list, train_label, val_list, val_label, test_list, test_label
    else:
        return train_list,train_label,test_list,test_label




def train_and_record_history(model, x_train, y_train, x_val, y_val, x_test, y_test, epochs=10):
    """
    Trains the CNN model, records metrics, and measures time for each epoch.

    Args:
        model: Compiled CNN model.
        x_train (ndarray): Training data.
        y_train (ndarray): Training labels.
        x_val (ndarray): Validation data.
        y_val (ndarray): Validation labels.
        x_test (ndarray): Test data.
        y_test (ndarray): Test labels.
        epochs (int): Number of epochs.

    Returns:
        dict: Training, validation, and test metrics, and epoch times.
    """
    epoch_times = []
    training_acc = []
    validation_acc = []
    test_acc = []
    training_recall = []
    validation_recall = []
    test_recall = []
    training_f1 = []
    validation_f1 = []
    test_f1 = []

    for epoch in range(epochs):
        start_time = time.time()

        # Train for one epoch
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=1, batch_size=32, verbose=1
        )

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # Store training and validation metrics
        training_acc.append(history.history['accuracy'][-1])
        validation_acc.append(history.history['val_accuracy'][-1])

        # Calculate training recall and F1-score
        y_train_pred = np.argmax(model.predict(x_train), axis=1)
        training_recall.append(recall_score(y_train, y_train_pred, average='weighted'))
        training_f1.append(f1_score(y_train, y_train_pred, average='weighted'))

        # Calculate validation recall and F1-score
        y_val_pred = np.argmax(model.predict(x_val), axis=1)
        validation_recall.append(recall_score(y_val, y_val_pred, average='weighted'))
        validation_f1.append(f1_score(y_val, y_val_pred, average='weighted'))

        # Evaluate on test set
        y_test_pred = np.argmax(model.predict(x_test), axis=1)
        test_acc_epoch = np.mean(y_test == y_test_pred)  # Test accuracy
        test_acc.append(test_acc_epoch)

        test_recall.append(recall_score(y_test, y_test_pred, average='weighted'))
        test_f1.append(f1_score(y_test, y_test_pred, average='weighted'))

        print(f"Epoch {epoch + 1}: {epoch_time:.2f} seconds, Test Accuracy: {test_acc_epoch:.4f}, Test Recall: {test_recall[-1]:.4f}, Test F1: {test_f1[-1]:.4f}")

    return {
        'epoch_times': epoch_times,
        'training_acc': training_acc,
        'validation_acc': validation_acc,
        'test_acc': test_acc,
        'training_recall': training_recall,
        'validation_recall': validation_recall,
        'test_recall': test_recall,
        'training_f1': training_f1,
        'validation_f1': validation_f1,
        'test_f1': test_f1
    }
def time_model(model, x_train, y_train, x_val, y_val, x_test, y_test, epochs=10):
    start_time = time.time()

    # Train for one epoch
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10, batch_size=32, verbose=1
    )

    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    return training_time

def plot_combined_metrics(metrics_small, metrics_large, filename_prefix):
    """
    Plots the accuracy trends of small and large CNN models in a single plot.

    Args:
        metrics_small (dict): Metrics for the small CNN model.
        metrics_large (dict): Metrics for the large CNN model.
        filename_prefix (str): Prefix for the saved plot filename.
    """
    epochs = range(1, len(metrics_small['training_acc']) + 1)

    plt.figure(figsize=(12, 8))

    # 28x28 CNN accuracies
    plt.plot(epochs, metrics_small['training_acc'], label='28x28 CNN - Training Accuracy', marker='o')

    plt.plot(epochs, metrics_small['test_acc'], label='28x28 CNN - Test Accuracy', marker='s')

    # 128x128 CNN accuracies
    plt.plot(epochs, metrics_large['training_acc'], label='128x128 CNN - Training Accuracy', marker='o', linestyle='--')
    plt.plot(epochs, metrics_large['test_acc'], label='128x128 CNN - Test Accuracy', marker='s', linestyle='--')

    # Add plot formatting
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy for 28x28 and 128x128 CNNs')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig('accuracy_trends_combined.png')
    plt.close()


def plot_combined_time(epoch_times_small, epoch_times_large, filename_prefix):
    """
    Plots the epoch-wise training times of small and large CNN models in a single plot.

    Args:
        epoch_times_small (list): Epoch-wise training times for the small CNN model.
        epoch_times_large (list): Epoch-wise training times for the large CNN model.
        filename_prefix (str): Prefix for the saved plot filename.
    """
    epochs = range(1, len(epoch_times_small) + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, epoch_times_small, label='28*28 CNN Training Time', marker='o')
    plt.plot(epochs, epoch_times_large, label='128*128 CNN Training Time', marker='x', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison: 28 vs. 128 CNN')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{filename_prefix}_combined_training_time.png')
    plt.close()


def plot_recall_f1(metrics_small, metrics_large, epochs):
    """
    Plots training, and test recall/F1-score for small and large CNN models.

    Args:
        metrics_small (dict): Metrics dictionary for small CNN model.
        metrics_large (dict): Metrics dictionary for large CNN model.
        epochs (int): Number of epochs.
    """
    epoch_range = range(1, epochs + 1)

    # Plot Recall
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_range, metrics_small['training_recall'], label='28x28 CNN - Training Recall', marker='o')
    plt.plot(epoch_range, metrics_small['validation_recall'], label='28x28 CNN - Validation Recall', marker='x')
    plt.plot(epoch_range, metrics_small['test_recall'], label='28x28 CNN - Test Recall', marker='s')
    plt.plot(epoch_range, metrics_large['training_recall'], label='128x128 CNN - Training Recall', marker='o', linestyle='--')
    plt.plot(epoch_range, metrics_large['validation_recall'], label='128x128 CNN - Validation Recall', marker='x', linestyle='--')
    plt.plot(epoch_range, metrics_large['test_recall'], label='128x128 CNN - Test Recall', marker='s', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall Comparison: Small vs Large CNN')
    plt.legend()
    plt.grid(True)
    plt.savefig('recall_comparison.png')
    plt.show()

    # Plot F1-Score
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_range, metrics_small['training_f1'], label='28x28 CNN - Training F1-Score', marker='o')
    plt.plot(epoch_range, metrics_small['validation_f1'], label='28x28 CNN - Validation F1-Score', marker='x')
    plt.plot(epoch_range, metrics_small['test_f1'], label='28x28 CNN - Test F1-Score', marker='s')
    plt.plot(epoch_range, metrics_large['training_f1'], label='128x128 CNN - Training F1-Score', marker='o', linestyle='--')
    plt.plot(epoch_range, metrics_large['validation_f1'], label='128x128 CNN - Validation F1-Score', marker='x', linestyle='--')
    plt.plot(epoch_range, metrics_large['test_f1'], label='128x128 CNN - Test F1-Score', marker='s', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title('F1-Score Comparison: Small vs Large CNN')
    plt.legend()
    plt.grid(True)
    plt.savefig('f1_comparison.png')
    plt.show()

def train_and_evaluate_model(model, train_data, train_labels, test_data, test_labels, epochs=10, batch_size=32):
    start_time = time.time()
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=0)
    training_time = time.time() - start_time
    loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
    return accuracy, training_time





################################################################################################################################



def create_mobileNet(input_shape, num_classes, num_fc_layers=1):
    """
    Creates and compiles a CNN model based on MobileNetV2.

    Args:
        input_shape (tuple): Shape of the input data (height, width, channels).
        num_classes (int): Number of output classes.
        num_fc_layers (int): Number of fully connected layers to add.

    Returns:
        model: Compiled MobileNetV2-based model.
    """
    # Load MobileNetV2 pre-trained on ImageNet
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze convolutional layers
    for layer in base_model.layers:
        layer.trainable = False

    fc_layers = []
    if num_fc_layers == 1:
        fc_layers = [layers.Dense(32, activation='relu')]
    elif num_fc_layers == 2:
        fc_layers = [layers.Dense(64, activation='relu'), layers.Dense(32, activation='relu')]
    elif num_fc_layers == 3:
        fc_layers = [layers.Dense(128, activation='relu'), layers.Dense(64, activation='relu'), layers.Dense(32, activation='relu')]
    elif num_fc_layers == 4:
        fc_layers = [layers.Dense(256, activation='relu'), layers.Dense(128, activation='relu'), layers.Dense(64, activation='relu'),
                             layers.Dense(32, activation='relu')]
    elif num_fc_layers == 5:
        fc_layers = [layers.Dense(256, activation='relu'),layers.Dense(256, activation='relu'), layers.Dense(128, activation='relu'), layers.Dense(64, activation='relu'),
                             layers.Dense(32, activation='relu')]
    elif num_fc_layers == 6:
        fc_layers = [layers.Dense(256, activation='relu'),layers.Dense(256, activation='relu'),layers.Dense(256, activation='relu'), layers.Dense(128, activation='relu'), layers.Dense(64, activation='relu'),
                             layers.Dense(32, activation='relu')]
    elif num_fc_layers == 7:
        fc_layers = [layers.Dense(256, activation='relu'),layers.Dense(256, activation='relu'),layers.Dense(256, activation='relu'),layers.Dense(256, activation='relu'), layers.Dense(128, activation='relu'), layers.Dense(64, activation='relu'),
                             layers.Dense(32, activation='relu')]



    # Add custom layers on top
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        *fc_layers,  # Add the fully connected layers based on num_fc_layers
        layers.Dense(num_classes, activation='softmax')  # Output layer for num_classes
    ])

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def prepare_data(size=128):
    """
    Prepares the training and test data for the model.

    Args:
        size (int): The size to which the images will be resized.

    Returns:
        tuple: Training and test data and labels.
    """
    train_dataset = OrganAMNIST(split="train", download=True, size=size)
    mean, std = compute_mean_std(train_dataset)

    train_dataset = OrganAMNIST(split="train", size=size, transform=lambda img: numpy_transform(img, mean, std))
    test_dataset = OrganAMNIST(split="test", size=size, transform=lambda img: numpy_transform(img, mean, std))

    train_list, train_label = convert_data_from_loader(train_dataset)
    test_list, test_label = convert_data_from_loader(test_dataset)

    # Reshape and convert to 3-channel images (for compatibility with MobileNetV2)
    train_list = train_list.reshape(-1, size, size, 1)
    train_list = np.repeat(train_list, 3, axis=-1)  # Convert grayscale to RGB
    train_label = np.array(train_label).reshape(-1)

    test_list = test_list.reshape(-1, size, size, 1)
    test_list = np.repeat(test_list, 3, axis=-1)
    test_label = np.array(test_label).reshape(-1)


    return train_list, train_label, test_list, test_label


def train_and_record_mbnt(model, train_list, train_label, test_list, test_label, epochs=10):
    """
    Trains the model and evaluates its performance.

    Args:
        model: The CNN model to train.
        train_list (ndarray): Training data.
        train_label (ndarray): Training labels.
        test_list (ndarray): Test data.
        test_label (ndarray): Test labels.
        epochs (int): Number of epochs.

    Returns:
        dict: Contains training time, test loss, and test accuracy.
    """
    training_acc = []
    test_acc = []
    start_time = time.time()
    for epoch in range(epochs):
        # Train for one epoch
        history = model.fit(train_list, train_label, epochs=1,validation_split=0.2, batch_size=32, verbose=1)


        # Store training and validation metrics
        training_acc.append(history.history['accuracy'][-1])
        y_test_pred = np.argmax(model.predict(test_list), axis=1)
        test_acc_epoch = np.mean(test_label == y_test_pred)  # Test accuracy
        test_acc.append(test_acc_epoch)

    total_time = time.time()-start_time
    loss, accuracy = model.evaluate(test_list, test_label, verbose=0)
    print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
    print(f"Training time: {total_time:.2f} seconds")
    test_predictions = model.predict(test_list)
    test_predictions = np.argmax(test_predictions, axis=1)

    test_recall = recall_score(test_label, test_predictions, average='weighted')
    test_f1 = f1_score(test_label, test_predictions, average='weighted')

    print(f"Test recall: {test_recall}")
    print(f"Test F1 score: {test_f1}")

    return {
        'train_acc': training_acc,
        'time': total_time,
        'test_loss': loss,
        'test_acc': test_acc,
        'test_recall': test_recall,
        'test_f1': test_f1

    }






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