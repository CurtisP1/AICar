import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import squeezenet1_1
import torch.quantization
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision.models import mobilenet_v3_small
from torchvision.models import squeezenet1_1
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import seaborn as sns
import time

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
    "turn right slightly": 4,
}

# Sequence length for temporal context
sequence_length = 5
frame_size = (64, 64)  # Reduced input size for faster processing
PIXEL_DIFF_THRESHOLD = 0.05

# Initialize counters for commands
command_counts = {cmd: 0 for cmd in command_mapping.keys()}

# Preprocessing function (with proper normalization for MobileNetV3)
def preprocess_frame(frame):
    try:
        resized_frame = cv2.resize(frame, frame_size)  # Resize to required dimensions
        normalized_frame = resized_frame.astype('float32') / 255.0  # Normalize to [0, 1]
        normalized_frame = (normalized_frame - 0.5) / 0.5  # Normalize to [-1, 1]
        return normalized_frame
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

# Data augmentation
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
])

# Adjust label for horizontal flips
def adjust_label_for_flip(label, was_flipped):
    if not was_flipped:
        return label
    if label == command_mapping["turn left"]:
        return command_mapping["turn right"]
    elif label == command_mapping["turn right"]:
        return command_mapping["turn left"]
    elif label == command_mapping["turn left slightly"]:
        return command_mapping["turn right slightly"]
    elif label == command_mapping["turn right slightly"]:
        return command_mapping["turn left slightly"]
    return label

def augment_frame_and_label(frame, label):
    frame = torch.tensor(frame).permute(2, 0, 1)  # Convert to tensor and permute to (C, H, W)
    was_flipped = False

    if torch.rand(1).item() < 0.5:  # 50% probability of flipping
        frame = transforms.functional.hflip(frame)
        was_flipped = True
        label = adjust_label_for_flip(label, was_flipped)

    frame = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)(frame)
    augmented_frame = frame.permute(1, 2, 0).numpy()
    return augmented_frame, label

# Dataset preparation
frames = []
labels = []
timestamps = []

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
    commands_data = commands_data.dropna(subset=["timestamp"]).drop(columns=["error"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_file}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1 / fps
    frame_count = 0
    video_frames = []
    video_labels = []
    last_frame = None
    discarded_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_timestamp = frame_count * frame_interval
        closest_command = commands_data.iloc[
            (commands_data["timestamp"] - current_timestamp).abs().idxmin()
        ]
        command_label = command_mapping.get(closest_command["command"], -1)

        if command_label == -1:
            frame_count += 1
            print(f"No valid command for timestamp {current_timestamp:.2f}. Skipping frame.")
            continue

        frame_preprocessed = preprocess_frame(frame)
        if frame_preprocessed is None:
            frame_count += 1
            print(f"Frame {frame_count} skipped during preprocessing.")
            continue

        frame_resized = cv2.resize(frame_preprocessed, frame_size)

        if last_frame is not None:
            pixel_diff = np.mean(np.abs(frame_resized - last_frame))
            if pixel_diff < PIXEL_DIFF_THRESHOLD:
                frame_count += 1
                print(f"Frame {frame_count} discarded due to low pixel difference.")
                discarded_frames += 1
                continue

        # Update processed command counts
        for cmd, idx in command_mapping.items():
            if command_label == idx:
                command_counts[cmd] += 1

        if np.random.rand() > 0.5:
            frame_resized, command_label = augment_frame_and_label(frame_resized, command_label)

        last_frame = frame_resized.copy()
        video_frames.append(frame_resized.astype('float32'))
        video_labels.append(command_label)
        frame_count += 1

    print(f"Discarded {discarded_frames} frames due to low pixel difference.")
    for i in range(len(video_frames) - sequence_length + 1):
        frames.append(video_frames[i:i + sequence_length])
        labels.append(video_labels[i + sequence_length - 2])
    cap.release()

# Print command counts after processing
print("Command Counts:")
for command, count in command_counts.items():
    print(f"{command}: {count}")

# Dataset definition
class RC_CarDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = np.array(frames, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int64)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        frame = torch.tensor(self.frames[idx]).permute(0, 3, 1, 2)
        label = torch.tensor(self.labels[idx])
        return frame, label

# Load dataset
dataset = RC_CarDataset(frames, labels)

# Check device in use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {device}")


# Compute class weights
labels_np = np.array(labels)  # Convert labels to NumPy array
class_weights = compute_class_weight(
    class_weight='balanced',  # Compute balanced weights
    classes=np.unique(labels_np),  # Unique class labels
    y=labels_np  # The labels
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print(f"Class Weights: {class_weights}")


train_size = int(0.8 * len(dataset))
indices = torch.randperm(len(dataset)).tolist()
train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

# Remaining code...



# Model definition
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


# Instantiate Model
model = OptimizedRC_CarModel(sequence_length).to(device)

# Training Configuration
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

# Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()


def check_validation_results(model, test_loader, device, command_mapping):
    """
    Evaluate the model on the test set and print detailed validation results.
    """
    model.eval()
    predictions = []
    ground_truth = []

    # Perform evaluation
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Ensure inputs and targets are moved to the correct device
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            # Collect predictions and ground truth
            predictions.extend(preds.cpu().numpy())  # Move predictions to CPU for processing
            ground_truth.extend(targets.cpu().numpy())  # Move ground truth to CPU

    # Count the occurrences of predictions and ground truth
    prediction_counts = Counter(predictions)
    ground_truth_counts = Counter(ground_truth)

    print("\n--- Validation Results ---")
    print("Prediction Distribution:")
    for cmd, idx in command_mapping.items():
        print(f"{cmd}: {prediction_counts[idx]}")

    print("\nGround Truth Distribution:")
    for cmd, idx in command_mapping.items():
        print(f"{cmd}: {ground_truth_counts[idx]}")

    # Generate a classification report
    command_names = list(command_mapping.keys())
    print("\n--- Classification Report ---")
    print(classification_report(ground_truth, predictions, target_names=command_names))

    # Generate a confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    print("\n--- Confusion Matrix ---")
    print(cm)

    # Visualize the confusion matrix using seaborn heatmap
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=command_names, yticklabels=command_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()
    except ImportError:
        print("Seaborn not installed. Skipping confusion matrix visualization.")


# Define early stopping parameters
early_stopping_patience = 5
best_val_loss = float("inf")
early_stopping_counter = 0
best_model_state = None
checkpoint_path = "squeeze_check.pth"
final_checkpoint_path = "squeeze_final.pth"

# Training Loop
for epoch in range(20):  # Number of epochs
    start_time = time.time()
    model.train()
    running_loss = 0.0  # Initialize running loss for this epoch
    correct_train = 0   # Initialize correct predictions for training
    total_train = 0     # Initialize total predictions for training

    # Training Phase
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # Mixed precision training
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

        # Calculate training accuracy
        preds = torch.argmax(outputs, dim=1)
        correct_train += (preds == targets).sum().item()
        total_train += targets.size(0)

    train_loss = running_loss / len(train_loader)  # Average loss for this epoch
    train_acc = correct_train / total_train  # Training accuracy

    # Validation Phase
    val_loss = 0.0  # Initialize validation loss
    correct_val = 0  # Initialize correct predictions for validation
    total_val = 0    # Initialize total predictions for validation
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()

            # Calculate validation accuracy
            preds = torch.argmax(outputs, dim=1)
            correct_val += (preds == targets).sum().item()
            total_val += targets.size(0)

    val_loss /= len(test_loader)  # Average validation loss
    val_acc = correct_val / total_val  # Validation accuracy

    # Scheduler Step
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']  # Log current learning rate

    # Save Best Model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        torch.save(best_model_state, checkpoint_path)
        print(f"Epoch {epoch + 1}: val_loss improved to {val_loss:.4f}, saving model to {checkpoint_path}")
        early_stopping_counter = 0
    else:
        print(f"Epoch {epoch + 1}: val_loss did not improve from {best_val_loss:.4f}")
        early_stopping_counter += 1

    # Epoch Summary
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch + 1}/{20}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s, Current LR: {current_lr:.6f}")

    # Early Stopping
    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered. Restoring the best model.")
        model.load_state_dict(best_model_state)
        break

# Save Final Model
torch.save(model.state_dict(), final_checkpoint_path)
print(f"Training complete. Final model saved as {final_checkpoint_path}")

# Check Validation Results
check_validation_results(model, test_loader, device, command_mapping)