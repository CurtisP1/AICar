import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, TimeDistributed, LSTM, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from collections import Counter
import matplotlib.pyplot as plt

# Set pandas options
pd.set_option('display.max_rows', None)

# Directories for videos and logs
video_dir = "videos/"
log_dir = "movement_logs/"

# Command mapping
command_mapping = {
    "forward": 0,
    "turn left": 1,
    "turn right": 2,
    "turn left slightly": 3,
    "turn right slightly": 4
}

# Initialize lists for frames, labels, and timestamps
frames = []
labels = []
timestamps = []

# Sequence length for temporal context
sequence_length = 5
frame_size = (128, 128)

# Initialize counters for commands
command_counts = {cmd: 0 for cmd in command_mapping}

# Pixel difference threshold
PIXEL_DIFF_THRESHOLD = 0.01

# Data augmentation helper
def augment_frame(frame):
    # Random horizontal flip
    if np.random.rand() < 0.5:
        frame = cv2.flip(frame, 1)
    # Random brightness adjustment
    if np.random.rand() < 0.5:
        brightness_factor = 1 + (np.random.rand() - 0.5) * 0.4
        frame = np.clip(frame * brightness_factor, 0, 1)
    # Random small rotation
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-10, 10)
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        frame = cv2.warpAffine(frame, rot_matrix, (frame.shape[1], frame.shape[0]))
    return frame

# Helper: Apply green masking to focus on green tape
def preprocess_with_green_mask(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(frame_hsv, (35, 50, 50), (85, 255, 255))  # Green range
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask_green)
    return masked_frame / 255.0  # Normalize

# Process multiple video-log pairs
for video_file in os.listdir(video_dir):
    if not video_file.endswith(".mp4"):
        continue

    log_file = os.path.join(log_dir, video_file.replace(".mp4", "_log.txt"))
    video_path = os.path.join(video_dir, video_file)

    if not os.path.exists(log_file):
        print(f"Log file missing for video: {video_file}")
        continue

    commands_data = pd.read_csv(log_file, sep="\t", skiprows=1, names=["timestamp", "command", "error"])
    commands_data["timestamp"] = pd.to_numeric(commands_data["timestamp"], errors="coerce")
    commands_data = commands_data.dropna(subset=["timestamp"])
    commands_data["timestamp"] = commands_data["timestamp"].astype(float)
    commands_data = commands_data.drop(columns=["error"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_file}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1 / fps
    frame_count = 0
    video_frames = []
    video_labels = []

    last_frame = None  # Initialize for pixel difference calculation
    discarded_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_timestamp = frame_count * frame_interval
        closest_command = commands_data.iloc[
            (commands_data["timestamp"].astype(float) - current_timestamp).abs().idxmin()
        ]
        command_label = command_mapping.get(closest_command["command"], -1)
        if command_label == -1:
            frame_count += 1
            continue

        command_counts[closest_command["command"]] += 1

        frame_preprocessed = preprocess_with_green_mask(frame)
        frame_resized = cv2.resize(frame_preprocessed, frame_size)

        # Apply data augmentation
        frame_augmented = augment_frame(frame_resized)

        # Calculate pixel difference if not the first frame
        if last_frame is not None:
            pixel_diff = np.mean(np.abs(frame_augmented - last_frame))
            if pixel_diff < PIXEL_DIFF_THRESHOLD:
                discarded_frames += 1
                frame_count += 1
                continue

        # Update last_frame and add current frame
        last_frame = frame_augmented.copy()
        video_frames.append(frame_augmented.astype('float32'))
        video_labels.append(command_label)
        timestamps.append(current_timestamp)

        frame_count += 1

    print(f"Discarded {discarded_frames} frames due to low pixel difference.")

    for i in range(len(video_frames) - sequence_length + 1):
        frames.append(video_frames[i:i + sequence_length])
        labels.append(video_labels[i + sequence_length - 2])

    cap.release()
    print(f"Processed frames for video {video_file}")

# Print command counts
print("Command counts:")
for command, count in command_counts.items():
    print(f"{command}: {count}")

# Convert lists to numpy arrays
X = np.array(frames, dtype='float32')
y = np.array(labels)

# Visualize preprocessed frames
for i in range(500):
    plt.imshow(frames[i][0])
    plt.title(f"Frame: {[i]}, Label: {y[i]}")
    plt.show()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Calculate class weights
class_counts = Counter(y_train)
class_weights = {cls: len(y_train) / (len(class_counts) * count) for cls, count in class_counts.items()}
print(class_counts)
print("Class weights:", class_weights)

# Define the model
model = Sequential([
    Input(shape=(sequence_length, frame_size[0], frame_size[1], 3)),
    TimeDistributed(MobileNetV2(include_top=False, weights='imagenet')),
    TimeDistributed(GlobalAveragePooling2D()),
    LSTM(128, activation='relu', return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dense(5, activation='softmax')  # Adjusted for 5 commands
])

model.layers[1].trainable = False

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ModelCheckpoint(filepath='model_temporal_augmented.keras', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
]

history = model.fit(
    tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(16).prefetch(tf.data.AUTOTUNE),
    validation_data=tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(16).prefetch(tf.data.AUTOTUNE),
    epochs=40,
    callbacks=callbacks,
    class_weight=class_weights
)

model.save('final_model_temporal_augmented.keras')
print("Model training complete and saved as 'final_model_temporal_augmented.keras'.")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
