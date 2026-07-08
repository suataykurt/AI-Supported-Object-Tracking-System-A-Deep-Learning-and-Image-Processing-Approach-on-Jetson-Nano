# AI-Supported-Object-Tracking-System-A-Deep-Learning-and-Image-Processing-Approach-on-Jetson-Nano

This project implements a real-time object tracking system powered by the **NVIDIA Jetson Nano**. It combines deep learning-based object detection with a precision pan-tilt servo mechanism.

## 📋 Project Overview
The system captures video via a USB camera, detects objects using the **YOLOv5s** algorithm, and tracks them by adjusting a 2-axis servo motor mechanism controlled by a **PCA9685** PWM driver board.

## 🛠 Hardware Components
* [cite_start]**Processing Unit:** NVIDIA Jetson Nano [cite: 50]
* [cite_start]**Detection Model:** YOLOv5s [cite: 14]
* [cite_start]**Motor Driver:** PCA9685 16-Channel PWM Driver [cite: 52]
* [cite_start]**Motion:** SG90 Pan-Tilt Servo Motors [cite: 53]

## 🚀 Key Features
* [cite_start]**Real-time Performance:** Capable of running at 18-22 FPS[cite: 193].
* [cite_start]**Optimization:** Utilizes CUDA-accelerated YOLOv5s for resource-constrained environments[cite: 127, 135].
* [cite_start]**Stable Control:** PCA9685 driver provides precise PWM signals for smooth motion[cite: 45].

## 📂 File Structure
* `code.py`: Main control script for object detection and servo movement.
* `report.pdf`: Detailed documentation including theoretical analysis, hardware setup, and experimental results.

## 💡 How to Run
1. Ensure your Jetson Nano is set up with PyTorch and OpenCV.
2. Connect the PCA9685 via I2C (SDA/SCL pins).
3. Install dependencies: `pip install adafruit-circuitpython-pca9685 opencv-python torch numpy`.
4. Run the code: `python code.py`.

---
*Developed by Refiye Deniz BATTAL and Suat AYKURT | Zonguldak Bülent Ecevit University*