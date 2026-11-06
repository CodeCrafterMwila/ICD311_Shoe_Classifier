# 👟 Intelligent Shoe Classifier

<p align="center">
  <img src="assets/banner.png" alt="CodeCrafterMwila AI Lab Banner" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Computer_Vision-005571?style=for-the-badge&logo=opencv&logoColor=white" alt="Computer Vision">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
</p>

A production-ready deep learning web application that identifies footwear categories using a custom-trained **MobileNetV3** convolutional neural network, powered by **PyTorch** and deployed via **Streamlit**.

---

## 🚀 Features
- **Real-Time AI Inference:** Instant classification using optimized MobileNetV3 small architecture.
- **Hardware Acceleration:** Automatic CUDA (GPU) detection with fallback to CPU execution.
- **Interactive Web UI:** Custom dark GitHub-inspired theme (`#0E1117`) with neon green accents and confidence metrics.
- **Top-K Predictions:** Displays the top category matches with probability breakdown bars.

---

## 🛠️ Project Structure

```text
ICD311_Shoe_Classifier/
├── .streamlit/         # Streamlit UI configuration
├── assets/             # Banner and visual graphics
├── models/             # Fine-tuned PyTorch model weights (.pth)
├── app.py              # Main Streamlit web application
├── shoe_classifier.ipynb # Model training & experimentation notebook
└── requirements.txt    # Python dependencies