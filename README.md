YogaMate 🧘‍♂️🎥
AI-Powered Smart Mirror for Real-Time Yoga Pose Correction

YogaMate is an intelligent fitness assistant that uses computer vision and deep learning to monitor and correct yoga postures in real time. Designed for Smart Mirrors, it helps users maintain correct form, reduce injury risk, and improve performance — all without needing a human trainer.

💡 Features
📷 Real-time video input from webcam

🔍 Human detection using CNN

🧍‍♀️ Pose estimation using MediaPipe (33 landmarks)

🧠 Pose classification with an Artificial Neural Network (ANN)

📐 Angle-based analysis for posture correctness

🔊 Voice feedback using pyttsx3 / pygame

🪞 Optimized for Smart Mirror environments

🏗️ System Architecture
Input Capture: Webcam feed split into frames.

Person Detection: CNN filters frames with no visible user.

Pose Estimation: MediaPipe detects 33 body landmarks.

Pose Recognition: Landmarks fed into an ANN for asana classification.

Form Correction: Critical angles calculated to evaluate posture.

Audio Feedback: Real-time voice guidance to improve form.

⚙️ Tech Stack
Pose Estimation: MediaPipe

Deep Learning: TensorFlow/Keras

Computer Vision: OpenCV

Audio Feedback: pyttsx3, pygame

Smart Mirror Integration: HDMI display + Webcam

Deployment: Streamlit (for interactive version)
