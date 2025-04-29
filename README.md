# CNN-Based Waste Classification Using BDWaste Dataset

## Project Overview

Waste management is a critical global challenge that demands innovative solutions. This project presents a deep learning-based system for automated waste classification, distinguishing between Digestive (biodegradable) and Indigestive (non-biodegradable) waste using the BDWaste dataset.

The project explores both custom Convolutional Neural Networks (CNNs) and Transfer Learning approaches, leveraging popular pre-trained models like VGG16, MobileNetV2, and ResNet50. Hyperparameter tuning and careful model evaluation were performed to optimize classification accuracy for real-world deployment.

## Dataset Description

- Source: BDWaste dataset (Mendeley Data)
- Total Images: 2,625
- Classes: 2 main classes (Digestive and Indigestive), with 21 subcategories
- Image Conditions: Captured in varied lighting and backgrounds using mobile devices

Digestive Waste Examples:
- Rice, Potato Peel, Fish Ash, Lemon Peel, Eggshell....etc

Indigestive Waste Examples:
- Plastic, Wire, Glass, Paper, Polythene...etc

## Methodology

- Data Preprocessing:
  - Images resized to 224x224 pixels
  - Normalized pixel values
  - Data augmentation applied only to training set

- Model Architectures:
  - Custom CNN (designed from scratch)
  - Tuned Pre-trained Models: VGG16, MobileNetV2, ResNet50

- Training Configuration:
  - Optimizer: Adam
  - Learning Rate: 0.0001
  - Loss Function: Categorical Crossentropy
  - Batch Size: 32
  - Epochs: 15
  - Early stopping applied to tuned pre-trained models only

- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix Analysis
  - Training and Validation Loss/Accuracy Curves

## Key Results

| Model                  | Test Accuracy | Precision | Recall | F1-Score |
|-------------------------|---------------|-----------|--------|----------|
| Custom CNN (Baseline)    | 85.92%         | 0.87      | 0.86   | 0.86     |
| Tuned Custom CNN         | 98.19%         | 0.98      | 0.98   | 0.98     |
| Tuned MobileNetV2        | 99.28%         | 0.99      | 0.99   | 0.99     |
| Tuned VGG16              | 92.78%         | 0.93      | 0.93   | 0.93     |
| Tuned ResNet50           | 68.23%         | 0.68      | 0.68   | 0.68     |

Best Performer: Tuned MobileNetV2 achieved nearly perfect classification across all evaluation metrics.

## Why Data Augmentation Only on Training Set

Data augmentation was applied exclusively to the training set to improve model generalization. Validation and test sets were kept original to ensure unbiased model evaluation.

## Project Highlights

- Designed and tuned a Custom CNN model
- Fine-tuned popular pre-trained models for the specific task
- Achieved over 99% test accuracy using Tuned MobileNetV2
- Visualized confusion matrices and training curves
- Applied regularization techniques like dropout and early stopping

## Future Work

- Expand the dataset with more waste categories
- Deploy lightweight models for real-time mobile applications
- Explore object detection models for multi-object waste classification
- Implement model compression for deployment on edge devices

## License

This project is for educational and research purposes only.

## Acknowledgements

- BDWaste Dataset (Mendeley Data)
- TensorFlow, Keras libraries
- Open-source deep learning community
