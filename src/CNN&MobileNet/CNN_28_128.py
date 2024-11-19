from utils_cnn_MobileNet import *
import numpy as np
from medmnist import OrganAMNIST
import os


def main():
    # Parameters
    epochs = 15
    num_classes = 11

    # Train and evaluate small CNN
    size_small = 28
    train_dataset_small = OrganAMNIST(split="train", size=size_small)
    mean_small, std_small = compute_mean_std(train_dataset_small)
    train_list_small, train_label_small, val_list_small, val_label_small, test_list_small, test_label_small = preprocess_data_cnn(
        size_small, mean_small, std_small
    )

    small_model = create_cnn_model(input_shape=(size_small, size_small, 1), num_classes=num_classes, is_large=False)
    metrics_small = train_and_record_history(small_model, train_list_small, train_label_small, val_list_small,
                                             val_label_small, test_list_small, test_label_small, epochs)
    # time_model(small_model, train_list_small, train_label_small, val_list_small, val_label_small, test_list_small, test_label_small, epochs)

    # Train and evaluate large CNN
    size_large = 128
    train_dataset_large = OrganAMNIST(split="train", size=size_large)
    mean_large, std_large = compute_mean_std(train_dataset_large)
    train_list_large, train_label_large, val_list_large, val_label_large, test_list_large, test_label_large = preprocess_data_cnn(
        size_large, mean_large, std_large
    )

    large_model = create_cnn_model(input_shape=(size_large, size_large, 1), num_classes=num_classes, is_large=True)
    # time_model(large_model, train_list_large, train_label_large, val_list_large, val_label_large, test_list_large, test_label_large, epochs)
    metrics_large = train_and_record_history(large_model, train_list_large, train_label_large, val_list_large,
                                             val_label_large, test_list_large, test_label_large, epochs)

    # Plot combined accuracy trends
    plot_combined_metrics(metrics_small, metrics_large, "comparison_cnn")

    # Plot combined training time trends
    plot_combined_time(metrics_small['epoch_times'], metrics_large['epoch_times'], "comparison_cnn")

    plot_recall_f1(metrics_small, metrics_large, epochs=15)


if __name__ == "__main__":
    main()