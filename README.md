# AI-Powered Lane Keeping System for Autonomous Cars

## 🚗 Introduction
Lane-keeping systems are critical in autonomous vehicles to ensure safe navigation and maintain the car's position within road lanes. This project demonstrates an AI-powered lane-keeping system using a camera and machine learning techniques.

---

## 📂 Table of Contents
1. [System Overview](#system-overview)
2. [Data Collection](#data-collection)
3. [Preprocessing](#preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Inference and Execution](#inference-and-execution)
7. [Challenges and Solutions](#challenges-and-solutions)
8. [Results](#results)
9. [Future Improvements](#future-improvements)
10. [Conclusion](#conclusion)

---

## ⚙️ System Overview

### **Components**
- **Pi Camera**: Captures real-time video of the road ahead.
- **AI Model**: Detects lane markings and predicts steering commands.
- **Raspberry Pi**: Executes the AI model and sends control signals.
- **Motor and Steering System**: Responds to AI commands for directional control.

---

## 📊 Data Collection

1. **Recording Videos**: Collect video data along a driving track.
2. **Annotation**: Label lane markings and corresponding steering commands.
3. **Data Augmentation**:
   - Adjust brightness, contrast, and saturation.
   - Add noise or perform horizontal flips.

---

## 🧹 Preprocessing

1. **Frame Preprocessing**:
   - Resize frames to fixed dimensions (e.g., 64x64).
   - Normalize pixel values to a range of [-1, 1].
2. **Masking**:
   - Use edge detection or color filtering for lane extraction.
   - Apply morphological operations to refine lane masks.

---

## 🏗️ Model Architecture

### **Feature Extraction**
- **Pretrained Backbone**: 
  - Example: [SqueezeNet](https://arxiv.org/abs/1602.07360) for lightweight feature extraction.
  - Extracts high-level features like lane boundaries.

### **Sequence Modeling**
- **LSTM (Long Short-Term Memory)**:
  - Processes video frame sequences.
  - Captures temporal dependencies for smooth lane tracking.

### **Fully Connected Layers**
- Outputs probabilities for steering commands:
  - **Forward**, **Turn Left**, **Turn Right**, etc.

---

## 🧠 Training

### **Loss Function**
- **Categorical Cross-Entropy**: Handles multi-class classification for steering commands.
- **Label Smoothing**: Prevents overconfidence in predictions.

### **Optimization**
- **Optimizer**: Adam with weight decay for regularization.
- **Learning Rate Scheduling**:
  - Warm-up scheduler for initial learning rate increase.
  - ReduceLROnPlateau for adaptive learning rate adjustments.

---

## 🖥️ Inference and Execution

1. **Real-Time Prediction**:
   - Capture frames continuously from the camera.
   - Preprocess and stack frames into a buffer.
   - Use the AI model for inference.
2. **Control System**:
   - Execute steering commands based on model predictions.
   - Adjust steering angles dynamically to maintain lane alignment.

---

## 🛠️ Challenges and Solutions

### **1. Lane Detection in Low Visibility**
- **Solution**: Train on diverse datasets and enhance preprocessing (e.g., adaptive thresholding).

### **2. Latency in Real-Time Execution**
- **Solution**: Optimize inference using model quantization and lightweight architectures.

### **3. Difficulty with Edge Cases**
- **Solution**: Add a wide angle lens to improve FOV allowing for better handling of edge cases.

---

## 📈 Results

### **Performance Metrics**
- **Accuracy**: Percentage of correct steering predictions around 90%.
- **Inference Time**: Average time for the AI model to process a frame around 60ms.
- **Lane Deviation**: Average distance from the center of the lane around 5cm.

---

## 🔮 Future Improvements

1. **Integration with GPS and IMU**:
   - Combine vision-based lane detection with sensor data for better accuracy.
2. **Dynamic Lane Adaptation**:
   - Extend the system to handle lane splits, merges, and curves.
3. **Edge AI Deployment**:
   - Deploy on optimized hardware like NVIDIA Jetson Nano or Coral Edge TPU.

---

## 🏁 Conclusion
This AI-powered lane-keeping system showcases the potential of computer vision in autonomous driving. Leveraging a camera and lightweight AI models, the system achieves real-time lane detection and steering, paving the way for safer and more efficient autonomous navigation.

---

