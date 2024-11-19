from utils_cnn_MobileNet import *


def experiment_num_layers():
    """
    Experiment with different numbers of fully connected layers on MobileNetV2.
    Evaluates training time, accuracy, recall, and F1 score.
    """
    # Load data
    size = 128  # Input size for the MobileNetV2 model
    train_list, train_label, test_list, test_label = prepare_data(size)

    # Results-CNN storage
    results = {}

    # Experiment with different FC layer configurations (1, 2, and 3 FC layers)
    for num_fc_layers in [1, 2, 3, 4, 5, 6, 7]:
        print(f"Training model with {num_fc_layers} fully connected layers...")
        model = create_mobileNet((size, size, 3), num_classes=11, num_fc_layers=num_fc_layers)

        # Train and evaluate the model
        metrics = train_and_record_mbnt(
            model,
            train_list,
            train_label,
            test_list,
            test_label,
            epochs=10
        )

        # Store results
        results[num_fc_layers] = metrics

    # Plot training accuracy, recall, and F1 score for different FC layer configurations
    plt.figure(figsize=(18, 6))

    # Training accuracy plot
    plt.subplot(1, 3, 1)
    for num_fc_layers, metrics in results.items():
        plt.plot(metrics['train_acc'], label=f'{num_fc_layers} FC layers')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    for num_fc_layers, metrics in results.items():
        plt.plot(metrics['test_acc'], label=f'{num_fc_layers} FC layers')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.bar(
        results.keys(),
        [metrics['time'] for metrics in results.values()],
        tick_label=[f'{num_fc_layers}layers' for num_fc_layers in results.keys()]
    )
    plt.title('Training Time')
    plt.ylabel('Time (seconds)')
    result_folder = "../Results-CNN-MobileNet"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Correcting the filename
    result_file = os.path.join(result_folder, "fc_layer_experiment_metrics.png")
    plt.savefig(result_file)
    plt.close()

    # Print results summary
    for num_fc_layers, metrics in results.items():
        print(f"\nResults-CNN for {num_fc_layers} FC layers:")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Test Recall: {metrics['test_recall']:.4f}")
        print(f"  Test F1 Score: {metrics['test_f1']:.4f}")
        print(f"  Training Time: {metrics['time']:.2f} seconds")


def train_test_acc_for2Dense():
    """
    Experiment with different numbers of fully connected layers on MobileNetV2.
    Evaluates training time, accuracy, recall, and F1 score.
    """
    # Load data
    size = 128  # Input size for the MobileNetV2 model
    train_list, train_label, test_list, test_label = prepare_data(size)

    # Results-CNN storage
    results = {}

    # Experiment with different FC layer configurations (1, 2, and 3 FC layers)

    model = create_mobileNet((size, size, 3), num_classes=11, num_fc_layers=2)

    # Train and evaluate the model
    metrics = train_and_record_mbnt(
        model,
        train_list,
        train_label,
        test_list,
        test_label,
        epochs=40
    )

    plot_performance(metrics, mb=True)


experiment_num_layers()
train_test_acc_for2Dense()