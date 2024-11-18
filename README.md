# ImageClassification-MultilayerPerceptrons-CNNs


## Overview

The complexity of modern medical imaging datasets has driven the need for advanced machine learning models capable of achieving accurate and efficient classification. This repository explores the OrganAMNIST dataset, a challenging benchmark derived from medical imagery, to evaluate the effectiveness of different neural network architectures for organ classification.

We investigate and compare the performance of:

Multilayer Perceptrons (MLPs): Foundational models in machine learning, which struggle with capturing spatial dependencies in image data.
Convolutional Neural Networks (CNNs): Specialized for spatial feature extraction, leveraging convolutional layers to create hierarchical representations.
Pre-trained MobileNetV2: A lightweight architecture that transfers knowledge from large-scale datasets, achieving high accuracy with reduced training time.
This project emphasizes the importance of hyperparameter optimization (e.g., learning rate, batch size, and regularization strength) and explores the trade-offs between computational efficiency and performance.

## Architecture 

MARC PLEASE WRITE!

## Key Features
Dataset:

- OrganAMNIST: A medical imaging dataset with grayscale images of 11 organs, resized to 28x28 pixels and 128x128 pixels for efficient experimentation.
- Dataset split includes training and test sets for rigorous evaluation.
- Normalization was done on the data using the Z-score technique seen in class.
  
Architectures Studied:

- MLPs: Baseline models that highlight the limitations of fully connected layers for image data.
- CNNs: Tested with various configurations (e.g., layer depth, kernel size, pooling strategies) to optimize performance.
- MobileNetV2: A pre-trained model offering high accuracy and efficiency.
  
Hyperparameter Optimization:
- Investigates learning rate, batch size, and regularization techniques (L1 and L2) to balance accuracy and computational cost.
  
Performance Metrics:

- Accuracy and Recall: For comparing model effectiveness.
- Training Time: For assessing computational efficiency.
- Robustness: Evaluating the model’s performance across different configurations.

## Applications

This repository provides insights into designing neural networks for medical image classification tasks. The findings are particularly relevant for:

- Developing robust, efficient AI models for healthcare applications.
- Exploring transfer learning for improving model accuracy on small, specialized datasets.
- Balancing computational cost with performance in resource-constrained environments.


## Results
### Performance Overview
- MLPs: Struggled to achieve high accuracy due to the lack of spatial feature extraction capabilities.
- CNNs: With optimal configurations, achieved 0.8325 accuracy, highlighting the importance of convolutional layers and pooling strategies.
- MobileNetV2: Achieved the best accuracy of 0.9249, showcasing the benefits of transfer learning for medical image classification.
Delivered a balance between accuracy and computational efficiency.
