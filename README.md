# Low-Light Face Recognition System

A computer vision system designed to improve face recognition accuracy in low-light environments using image enhancement and deep learning.
Technologies: Python, OpenCV, TensorFlow, MobileNetV2

## Problem Statement

Face recognition systems perform poorly in low-light environments due to noise, low contrast, and poor illumination. This project aims to enhance facial images captured in low-light conditions and improve recognition accuracy using deep learning.

##Methodology

The system follows the pipeline below:

1. Dataset creation using webcam
2. Face detection and cropping
3. Low-light enhancement using CLAHE and gamma correction
4. Data preprocessing and augmentation
5. Hybrid deep learning model training (MobileNetV2 + CNN)
6. Real-time face recognition

## System Architecture

Dataset Capture
        ↓
Face Detection
        ↓
Low-Light Enhancement
(CLAHE + Gamma Correction)
        ↓
Image Preprocessing
        ↓
Hybrid Deep Learning Model
(MobileNetV2 + CNN)
        ↓
Real-Time Face Recognition

## Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- Scikit-learn

## Project Structure

```
low-light-face-recognition
│
├── dataset_generation.py
├── preprocessing.py
├── hybrid_model.py
├── realtime_recognition.py
├── requirements.txt
└── README.md
```

dataset_generation.py → captures face images from webcam  
preprocessing.py → performs low-light enhancement and dataset preparation  
hybrid_model.py → trains the hybrid MobileNetV2 + CNN model  
realtime_recognition.py → performs real-time face recognition using webcam

## Research Work

This project is part of a research paper titled:

"Face Recognition in Low Light Conditions using Hybrid CNN Models"

Accepted at: ETFI, 2026  
The work proposes a hybrid CNN architecture combined with low-light enhancement techniques for improved face recognition performance.

## Future Improvements

- Improve robustness in extremely dark environments
- Use more advanced face detectors such as MTCNN
- Train on a larger dataset
- Deploy the model as a web application

## Results

The system successfully detects and recognizes faces in low-light environments after enhancement.

Academic Project – B.Tech CSE AI & Data Science  
MIT World Peace University
