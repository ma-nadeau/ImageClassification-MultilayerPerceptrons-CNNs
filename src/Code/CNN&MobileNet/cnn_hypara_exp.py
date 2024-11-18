from utils_cnn_MobileNet import *
from medmnist import OrganAMNIST


def experiment_hyperparameters(input_size):
    # Load data

    train_dataset = OrganAMNIST(split="train", size=input_size)
    mean, std = compute_mean_std(train_dataset)
    num_classes = 11
    train_data, train_labels, test_data, test_labels = preprocess_data_cnn(input_size, mean, std, validation=False)

    # Hyperparameter combinations
    filter_sizes = [8, 16, 32]
    kernel_sizes = [(3, 3), (5, 5)]
    strides = [(1, 1), (2, 2)]
    paddings = ['valid', 'same']

    results = []

    # Experiment loop
    for num_filters in filter_sizes:
        for kernel_size in kernel_sizes:
            for stride in strides:
                for padding in paddings:
                    print(
                        f"Testing filters={num_filters}, kernel_size={kernel_size}, stride={stride}, padding={padding}")
                    if input_size == 28:
                        model = create_cnn_model((input_size, input_size, 1), num_classes,
                                                 (num_filters, num_filters * 2, num_filters * 4), kernel_size, stride,
                                                 padding, is_large=False)
                    else:
                        model = create_cnn_model((input_size, input_size, 1), num_classes,
                                                 (num_filters, num_filters * 2, num_filters * 4), kernel_size, stride,
                                                 padding, is_large=True)
                    accuracy, training_time = train_and_evaluate_model(model, train_data, train_labels, test_data,
                                                                       test_labels)
                    results.append({
                        'filters': num_filters,
                        'kernel_size': kernel_size,
                        'stride': stride,
                        'padding': padding,
                        'accuracy': accuracy,
                        'training_time': training_time
                    })

    # Display and visualize results
    results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    for result in results:
        print(result)

    # Extracting data
    training_times = [result['training_time'] for result in results]
    accuracies = [result['accuracy'] for result in results]
    filters = [result['filters'] for result in results]
    labels = [
        f"F:{result['filters']}, K:{result['kernel_size']}, S:{result['stride']}, P:{result['padding']}"
        for result in results]

    # Plot
    plt.figure(figsize=(14, 7))
    scatter = plt.scatter(
        training_times, accuracies,
        c=filters, cmap='viridis', s=100, edgecolor='k', alpha=0.8
    )
    plt.colorbar(scatter, label='Number of Filters')
    plt.title('Comparison of Training Time vs Accuracy')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Adding shortened labels with slight offsets
    for i, label in enumerate(labels):
        plt.text(
            training_times[i] + 0.1,  # Offset in x-direction
            accuracies[i] + 0.0005,  # Offset in y-direction
            label,
            fontsize=8,
            ha='left',
            va='bottom'
        )

    plt.tight_layout()
    plt.savefig('Comparison_through_CNNs.png')


experiment_hyperparameters(128)