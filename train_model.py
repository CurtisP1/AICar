import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# Load the dataset
data = pd.read_csv("pwm_log.txt", header=None, names=["timestamp", "image_path", "command"])

# Preprocess images and labels
images = []
labels = []

for _, row in data.iterrows():
    img = cv2.imread(row["image_path"])
    img = cv2.resize(img, (64, 64))  # Resize images for the model
    img = img / 255.0  # Normalize pixel values
    images.append(img)

    # Map commands to numerical values
    if row["command"] == "forward":
        labels.append(0)
    elif row["command"] == "turn_left":
        labels.append(1)
    elif row["command"] == "turn_right":
        labels.append(2)

X = np.array(images)
y = np.array(labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Output: 3 classes (0: forward, 1: turn_left, 2: turn_right)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save('line_follower_model.h5')
print("Model training complete and saved as 'line_follower_model.h5'.")
