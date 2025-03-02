# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FG2pvyDul8ZVc-xEJE1stsig2OxjDPQH
"""

from google.colab import drive
import zipfile
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define dataset path
dataset_zip_path = "/content/drive/My Drive/BDWaste.zip"  # Update this path if necessary
extract_path = "/content/BDWaste"

# Extract dataset
with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset extracted successfully!")

import os

# Define dataset path
dataset_inner_path = "/content/BDWaste/BDWaste"

# Check class folders
print("Dataset Structure:")
for category in os.listdir(dataset_inner_path):
    category_path = os.path.join(dataset_inner_path, category)
    if os.path.isdir(category_path):
        print(f"📂 {category} → {len(os.listdir(category_path))} subfolders")

# Check inside one sample subfolder
sample_subfolder = os.path.join(dataset_inner_path, "Biodegradable")  # Update as needed
if os.path.isdir(sample_subfolder):
    print(f"\n🔍 Sample subfolder: {sample_subfolder}")
    print(f"Contents: {os.listdir(sample_subfolder)[:5]}")  # Show only first 5

import os
import shutil
import random

# Define dataset paths
dataset_inner_path = "/content/BDWaste/BDWaste"
test_folder = "/content/BDWaste/Test"

# Define class folders
digestive_src = os.path.join(dataset_inner_path, "Digestive")
indigestive_src = os.path.join(dataset_inner_path, "Indigestive")

digestive_dst = os.path.join(test_folder, "Digestive")
indigestive_dst = os.path.join(test_folder, "Indigestive")

# Ensure test directories exist
os.makedirs(digestive_dst, exist_ok=True)
os.makedirs(indigestive_dst, exist_ok=True)

# Function to find and move 10% of images from sub-subfolders
def move_images_to_test(src_class_folder, dst_class_folder, percentage=10):
    all_images = []

    # Loop through all subfolders and collect images
    for subfolder in os.listdir(src_class_folder):
        subfolder_path = os.path.join(src_class_folder, subfolder)
        if os.path.isdir(subfolder_path):  # Only process folders
            for inner_subfolder in os.listdir(subfolder_path):  # Go deeper
                inner_subfolder_path = os.path.join(subfolder_path, inner_subfolder)
                if os.path.isdir(inner_subfolder_path):
                    images = [os.path.join(inner_subfolder_path, f)
                              for f in os.listdir(inner_subfolder_path)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.extend(images)

    # Move 10% of images
    num_to_move = int(len(all_images) * (percentage / 100))
    images_to_move = random.sample(all_images, min(num_to_move, len(all_images)))

    for img_path in images_to_move:
        img_name = os.path.basename(img_path)
        shutil.move(img_path, os.path.join(dst_class_folder, img_name))  # Move to correct test folder

# Move images to the Test set
move_images_to_test(digestive_src, digestive_dst)
move_images_to_test(indigestive_src, indigestive_dst)

print("\n✅ Test Set (10%) successfully created with images in correct folders!")

print(f"Digestive (Test Set): {len(os.listdir(digestive_dst))} images")
print(f"Indigestive (Test Set): {len(os.listdir(indigestive_dst))} images")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define test image generator (No Augmentation)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load test set
test_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # No shuffling for test set
)

print(f"✅ Test images loaded successfully: {test_generator.samples}")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Define dataset paths
dataset_inner_path = "/content/BDWaste/BDWaste"  # Main dataset (Biodegradable, Indigestive)
test_folder = "/content/BDWaste/Test"  # Separate test dataset

# ✅ Apply data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% Train, 10% Validation
)

# ✅ Load Training Data (80%)
train_generator = train_datagen.flow_from_directory(
    dataset_inner_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

# ✅ Load Validation Data (10%)
val_generator = train_datagen.flow_from_directory(
    dataset_inner_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# ✅ Load Test Data (10%) - No Augmentation
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # No shuffling for test set
)

# ✅ Print dataset summary
print(f"\n✅ Dataset Split:")
print(f"Training Images: {train_generator.samples}")
print(f"Validation Images: {val_generator.samples}")
print(f"Test Images: {test_generator.samples}")

import numpy as np
import matplotlib.pyplot as plt

# ✅ Get a batch of augmented images
sample_batch, sample_labels = next(train_generator)

# ✅ Display first 5 augmented images
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(sample_batch[i])
    plt.axis('off')
plt.show()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ✅ Define a simple CNN model
def create_basic_cnn(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Prevent overfitting
        Dense(1, activation='sigmoid')  # Binary classification (Biodegradable vs Non-Biodegradable)
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# ✅ Create model
basic_cnn = create_basic_cnn()

# ✅ Train the model
history = basic_cnn.fit(
    train_generator,
    epochs=10,  # Adjust as needed
    validation_data=val_generator
)

# ✅ Save the model
basic_cnn.save('/content/drive/MyDrive/basic_cnn_model.h5')

print("\n✅ Basic CNN Model Training Complete!")

import matplotlib.pyplot as plt

# ✅ Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")

    plt.show()

# ✅ Show accuracy & loss curves
plot_training_history(history)
