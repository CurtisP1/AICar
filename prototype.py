import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# Load the steering commands log
commands_data = pd.read_csv("pwm_commands/pwm_log_20241017-155225.txt", header=None, names=["timestamp", "command"])

# Map commands to numerical labels
command_mapping = {"forward": 0, "turn_left": 1, "turn_right": 2}

# Preprocess video frames and associate them with commands
video_path = "video.mp4"  # Path to the video file
frames = []
labels = []

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
frame_interval = 1 / fps  # Time between frames in seconds

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate the current frame's timestamp
    current_timestamp = frame_count * frame_interval
    frame_count += 1

    # Find the closest command timestamp
    closest_command = commands_data.iloc[
        (commands_data["timestamp"].astype(float) - current_timestamp).abs().idxmin()
    ]

    # Resize and normalize the frame
    frame_resized = cv2.resize(frame, (64, 64))  # Resize frames for the model
    frame_normalized = frame_resized / 255.0  # Normalize pixel values

    # Append the frame and its corresponding command
    frames.append(frame_normalized)
    labels.append(command_mapping.get(closest_command["command"], -1))  # Default to -1 for unknown commands

# Convert lists to numpy arrays
X = np.array(frames)
y = np.array(labels)

# Filter out frames with invalid commands
valid_indices = y != -1
X = X[valid_indices]
y = y[valid_indices]

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
