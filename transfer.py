import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import squeezenet1_1
from sklearn.utils.class_weight import compute_class_weight
import time
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import pandas as pd
import os
import cv2

# Command mapping
command_mapping = {
    "forward": 0,
    "turn left": 1,
    "turn right": 2,
    "turn left slightly": 3,
    "turn right slightly": 4,
}

# Preprocessing function
def preprocess_frame(frame, frame_size=(64, 64)):
    """Resize and normalize frame."""
    resized_frame = cv2.resize(frame, frame_size)
    normalized_frame = resized_frame.astype('float32') / 255.0
    normalized_frame = (normalized_frame - 0.5) / 0.5  # Normalize to [-1, 1]
    return normalized_frame

# Enhanced Data Augmentation
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.6, 1.0)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
])

# Gradual Layer Unfreezing
def unfreeze_layers(model, layers_to_unfreeze):
    for name, param in model.squeezenet.named_parameters():
        if any(layer in name for layer in layers_to_unfreeze):
            param.requires_grad = True
            print(f"Unfroze layer: {name}")

# Specify the layers to unfreeze for SqueezeNet
unfrozen_layers = ["fire8", "fire9"]  # Last two fire modules for fine-tuning

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


# Dataset definition
class RC_CarDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if frame.ndim == 3:  # Single frame
            frame = torch.tensor(frame).permute(2, 0, 1)  # Rearrange to (channels, height, width)
        elif frame.ndim == 4:  # Sequence of frames
            frame = torch.tensor(frame).permute(0, 3, 1, 2)  # Rearrange to (sequence_length, channels, height, width)
        else:
            raise ValueError(f"Unexpected frame dimensions: {frame.shape}")

        label = torch.tensor(self.labels[idx])
        return frame, label


# Process new track videos
def process_new_track_videos(video_dir, log_dir, frame_size=(64, 64), pixel_diff_threshold=0.05, sequence_length=5):
    """
    Process videos and their corresponding log files to generate frame sequences and labels.

    Parameters:
        video_dir (str): Directory containing video files.
        log_dir (str): Directory containing corresponding log files.
        frame_size (tuple): Target size for resizing frames (width, height).
        pixel_diff_threshold (float): Threshold for detecting significant frame changes.
        sequence_length (int): Length of frame sequences for LSTM input.

    Returns:
        np.array: Sequences of frames.
        np.array: Corresponding labels for each sequence.
    """
    new_frames = []
    new_labels = []

    for video_file in os.listdir(video_dir):
        if not video_file.endswith(".mp4"):
            continue

        log_file = os.path.join(log_dir, video_file.replace(".mp4", "_log.txt"))
        video_path = os.path.join(video_dir, video_file)

        if not os.path.exists(log_file):
            print(f"Log file missing for video: {video_file}")
            continue

        # Load command logs
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
                continue

            # Preprocess the frame
            frame_preprocessed = preprocess_frame(frame, frame_size)
            if frame_preprocessed is None:
                frame_count += 1
                continue

            # Discard low-motion frames
            if last_frame is not None:
                pixel_diff = np.mean(np.abs(frame_preprocessed - last_frame))
                if pixel_diff < pixel_diff_threshold:
                    frame_count += 1
                    print(f"Frame {frame_count} discarded due to low pixel difference.")
                    discarded_frames += 1
                    continue

            # Apply data augmentation to 50% of frames
            if np.random.rand() > 0.5:
                frame_preprocessed, command_label = augment_frame_and_label(frame_preprocessed, command_label)

            last_frame = frame_preprocessed
            video_frames.append(frame_preprocessed.astype('float32'))
            video_labels.append(command_label)
            frame_count += 1

        print(f"Discarded {discarded_frames} frames due to low pixel difference.")
        # Generate sequences for LSTM input
        for i in range(len(video_frames) - sequence_length + 1):
            new_frames.append(video_frames[i:i + sequence_length])
            new_labels.append(video_labels[i + sequence_length - 1])
        cap.release()

    print(f"Processed {len(new_frames)} sequences from videos.")
    return np.array(new_frames, dtype=np.float32), np.array(new_labels, dtype=np.int64)

def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)

# Label Smoothing Loss Function
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, class_weights=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, logits, target):
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        # Convert targets to one-hot encoding
        target_one_hot = torch.zeros_like(logits).scatter(1, target.unsqueeze(1), 1)
        weight = target_one_hot * (1 - self.smoothing) + self.smoothing / logits.size(-1)

        if self.class_weights is not None:
            # Broadcast class weights to match weight shape
            weight = weight * self.class_weights.view(1, -1)

        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

# Load pretrained model
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


# Load pre-trained model
sequence_length = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OptimizedRC_CarModel(sequence_length).to(device)
pretrained_path = "squeeze_final.pth"
model.load_state_dict(torch.load(pretrained_path))
print("Pretrained model loaded successfully.")

# Freeze lower layers of SqueezeNet
for param in model.squeezenet.parameters():
    param.requires_grad = False

# Process new track videos
new_video_dir = "new_videos/"
log_dir = "new_logs/"
new_frames, new_labels = process_new_track_videos(new_video_dir, log_dir)
print(f"Processed {len(new_frames)} frames from new track.")

# Add command counts
command_counts = Counter(new_labels)
print("Command Distribution in New Dataset:")
for command, count in command_counts.items():
    print(f"{list(command_mapping.keys())[list(command_mapping.values()).index(command)]}: {count}")

# Create new dataset and dataloaders
new_dataset = RC_CarDataset(new_frames, new_labels)
train_size = int(0.8 * len(new_dataset))
indices = torch.randperm(len(new_dataset)).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(new_dataset, train_indices)
val_dataset = Subset(new_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)  # Define test_loader

# Compute class weights
labels_np = np.array(new_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_np), y=labels_np)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Class Weights: {class_weights}")


train_dataset = Subset(new_dataset, train_indices)
val_dataset = Subset(new_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# Fine-tuning setup
criterion = LabelSmoothingCrossEntropy(smoothing=0.1, class_weights=class_weights_tensor)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)

num_epochs = 100
num_training_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * num_training_steps)  # Warm up for 10% of training
warmup_scheduler = get_lr_scheduler(optimizer, warmup_steps, num_training_steps)
reduce_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6, verbose = True)

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
checkpoint_path = "last_check.pth"
final_checkpoint_path = "main_squeeze.pth"

# Training Loop
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Unfreeze layers after epoch 10
    if epoch == 10:
        unfreeze_layers(model, unfrozen_layers)

    # Training Phase
    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Mixed Precision Training
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Warmup Scheduler Step (only active for first 10 epochs)
        if epoch < 10:
            warmup_scheduler.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct_train += (preds == targets).sum().item()
        total_train += targets.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train

    # Validation Phase
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            preds = torch.argmax(outputs, dim=1)
            correct_val += (preds == targets).sum().item()
            total_val += targets.size(0)

    val_loss /= len(test_loader)
    val_acc = correct_val / total_val

    # Scheduler Step (Use ReduceLROnPlateau only after warmup)
    if epoch >= 10:
        reduce_lr_scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr']

    # Save Best Model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        torch.save(best_model_state, checkpoint_path)
        early_stopping_counter = 0
        print(f"Epoch {epoch + 1}: val_loss improved to {val_loss:.4f}, saving model to {checkpoint_path}")
    else:
        early_stopping_counter += 1
        print(f"Epoch {epoch + 1}: val_loss did not improve from {best_val_loss:.4f}")

    # Epoch Summary
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, "
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