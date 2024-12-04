import os
import torch
import torch.nn as nn
import torch.quantization  # Import for quantization
from torchvision.models import mobilenet_v3_small
from torchvision.models import squeezenet1_1
import cv2
import numpy as np
from picamera2 import Picamera2
from time import sleep
from picarx import Picarx
from collections import deque
import time

# Initialize car control and camera
px = Picarx()
camera = Picamera2()
camera.configure(camera.create_preview_configuration(main={"size": (640, 480)}))
camera.start()

# Ensure you're running on CPU
device = torch.device("cpu")

# Define command_mapping before the model class
command_mapping = {
    "forward": 0,
    "turn left": 1,
    "turn right": 2,
    "turn left slightly": 3,
    "turn right slightly": 4,
}

# Updated Model Definition
class OptimizedRC_CarModel(nn.Module):
    def __init__(self, sequence_length):
        super().__init__()
        # Load SqueezeNet and extract features
        squeezenet = squeezenet1_1(weights="IMAGENET1K_V1")
        self.squeezenet = nn.Sequential(*list(squeezenet.features.children()))

        # Adaptive pooling to match LSTM input size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM layers
        self.lstm = nn.LSTM(512, 128, batch_first=True, num_layers=1, dropout=0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, len(command_mapping))
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()

        # Flatten sequence and pass through SqueezeNet feature extractor
        x = x.view(-1, channels, height, width)
        x = self.squeezenet(x)

        # Adaptive pooling and reshaping for LSTM
        x = self.pool(x).view(batch_size, seq_len, -1)

        # Pass through LSTM
        x, _ = self.lstm(x)

        # Fully connected layers
        x = self.relu(self.fc1(x[:, -1, :]))
        return self.fc2(x)


# Load the trained model
print("Loading trained PyTorch model...")
sequence_length = 5
model = OptimizedRC_CarModel(sequence_length)

# Path to the trained model file
model_path = "../../../../../Downloads/main_squeeze.pth"

# Check if the trained model file exists
if not os.path.exists(model_path):
    print(f"Model file {model_path} does not exist. Exiting...")
    exit()

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Quantize the model dynamically
print("Applying dynamic quantization to the model...")
model = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
)
print("Quantization applied successfully.")

# Set model to evaluation mode
model.to(device)
model.eval()

# Updated Preprocessing Function
def preprocess_frame(frame):
    """
    Resize and normalize the frame to match the training preprocessing pipeline.
    Ensure the frame has 3 channels (RGB).
    """
    try:
        # Ensure frame has 3 channels (convert RGBA to RGB if necessary)
        if frame.shape[2] == 4:  # Check if there's an alpha channel
            frame = frame[:, :, :3]  # Drop the alpha channel

        # Resize and normalize the frame
        resized_frame = cv2.resize(frame, (64, 64))  # Match training dimensions
        normalized_frame = resized_frame.astype("float32") / 255.0  # Normalize pixel values to [0, 1]
        normalized_frame = (normalized_frame - 0.5) / 0.5  # Normalize to [-1, 1]
        return normalized_frame
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None


# Frame buffer for sequential input
frame_buffer = deque(maxlen=sequence_length)

# Buffer for smoothing predictions
prediction_buffer = deque(maxlen=5)  # Use the last 5 predictions for smoothing


def smooth_predictions(prediction):
    """
    Smooth predictions using a majority vote or average.
    """
    global prediction_buffer
    prediction_buffer.append(prediction)

    # Majority vote
    smoothed_prediction = max(set(prediction_buffer), key=prediction_buffer.count)
    return smoothed_prediction


def predict_command(frame):
    """
    Predict the command based on the input frame sequence using the quantized model.
    """
    global frame_buffer

    # Preprocess the current frame
    frame_preprocessed = preprocess_frame(frame)

    if frame_preprocessed is None:
        print("Frame preprocessing failed. Skipping frame.")
        return None

    # Convert frame to a PyTorch tensor
    frame_tensor = torch.tensor(frame_preprocessed).permute(2, 0, 1)  # Shape: [channels, height, width]

    # Add the preprocessed frame to the frame buffer
    frame_buffer.append(frame_tensor)

    # Check if the buffer is filled
    if len(frame_buffer) < sequence_length:
        print(f"Not enough frames in buffer ({len(frame_buffer)}/{sequence_length}). Waiting for buffer to fill.")
        return None

    # Stack frames in the buffer
    input_tensor = torch.stack(list(frame_buffer), dim=0)  # Shape: [sequence_length, channels, height, width]
    input_tensor = input_tensor.unsqueeze(0).to(device)  # Shape: [1, sequence_length, channels, height, width]

    # Perform inference
    try:
        start_time = time.time()  # Start timing
        with torch.no_grad():
            predictions = model(input_tensor)  # Shape: [1, num_classes]
        inference_time = time.time() - start_time  # Calculate inference time
        print(f"Inference Time: {inference_time:.4f} seconds")  # Print the inference time
    except Exception as e:
        print(f"Inference error: {e}")
        return None

    # Convert predictions to numpy for further processing
    predictions = predictions.cpu().numpy()

    # Confidence threshold
    max_prob = np.max(predictions)
    if max_prob < 0.7:
        print("Low confidence in prediction, skipping.")
        return -1

    command = np.argmax(predictions)
    smoothed_command = smooth_predictions(command)
    print(f"Predicted command: {command}, Smoothed command: {smoothed_command}, Confidence: {max_prob:.4f}")
    return smoothed_command

current_camera_angle = 0  # Initialize the camera angle

def smooth_camera_angle(target_angle, step=5):
    """
    Smoothly adjusts the camera angle to the target angle in small increments.
    :param target_angle: The desired final angle for the camera.
    :param step: The incremental step for each adjustment.
    """
    global current_camera_angle
    while abs(current_camera_angle - target_angle) > step:
        if current_camera_angle < target_angle:
            current_camera_angle += step
        elif current_camera_angle > target_angle:
            current_camera_angle -= step
        px.set_cam_pan_angle(current_camera_angle)
        sleep(0.05)  # Small delay for smooth movement
    # Final adjustment to ensure exact target angle
    current_camera_angle = target_angle
    px.set_cam_pan_angle(current_camera_angle)


def set_camera_servo_angle(angle):
    """
    Smoothly sets the camera servo to the desired angle using smooth_camera_angle.
    :param angle: The desired angle for the camera.
    """
    try:
        smooth_camera_angle(angle)
    except Exception as e:
        print(f"Error setting camera servo angle: {e}")


def execute_command(command, speed):
    """
    Execute the predicted command by controlling the car's movement
    and synchronizing the camera's angle with the steering angle.
    """
    if command == -1:  # Stop command
        px.stop()
        return

    if command == 0:  # Forward
        px.set_dir_servo_angle(0)
        px.forward(speed)
        set_camera_servo_angle(0)
    elif command == 1:  # Turn Left
        px.set_dir_servo_angle(-25)
        px.forward(speed)
        set_camera_servo_angle(-15)  # Slightly align camera to the left
    elif command == 2:  # Turn Right
        px.set_dir_servo_angle(25)
        px.forward(speed)
        set_camera_servo_angle(15)  # Slightly align camera to the right
    elif command == 3:  # Slight Turn Left
        px.set_dir_servo_angle(-15)
        px.forward(speed)
        set_camera_servo_angle(-15)  # Align camera slightly left
    elif command == 4:  # Slight Turn Right
        px.set_dir_servo_angle(15)
        px.forward(speed)
        set_camera_servo_angle(15)  # Align camera slightly right
    else:
        px.stop()


try:
    print("Starting AI-controlled car with PyTorch. Press Ctrl+C to stop.")
    speed = 10  # Default speed

    while True:
        frame = camera.capture_array()  # Capture frame
        smoothed_command = predict_command(frame)  # Predict smoothed command

        if smoothed_command is not None:  # Only execute if a command is available
            execute_command(smoothed_command, speed)
        else:
            print("Waiting for sufficient frames in buffer...")

except KeyboardInterrupt:
    print("Program terminated by user.")
finally:
    px.stop()
    camera.stop_preview()
    camera.stop()
