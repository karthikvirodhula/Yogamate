import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import pygame
import pyttsx3
import au_functions
import os
import threading

pygame.mixer.init()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load Model & Encoder
model = load_model("model.h5")
with open('onehot_encoder_output.pkl', "rb") as file:
    oe = pickle.load(file)

st.title("TrainWithAI - Surya Namaskar")

lang = st.sidebar.selectbox("Select Language", ['English', 'Telugu', 'Tamil', 'Hindi', 'Kannada'])
run = st.button('Start')

# Pose Dictionary
poses = {
    1: "pranamasana", 2: "hasta uttanasana", 3: "Hastapadasana", 4: "Ashwa Sanchalanasana left",
    5: "stickpose", 6: "Ashtanga namaskara", 7: "cobra", 8: "adho mukha svanasana", 9: "Ashwa Sanchalanasana right", 0: "Nothing"
}

count = 1
curr_pos = 1
status = ''

# Stream URL
stream_url = 'http://192.168.1.63:8080/?action=stream'
cap = cv2.VideoCapture(0)

# Streamlit UI Elements
FRAME_WINDOW = st.image([])
accuracy_widget = st.empty()


def play_audio(file_path):
    """Preload and play audio for faster response"""
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


def capture_frame():
    """Thread function to capture frames asynchronously"""
    global frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break


# Start video capture in a separate thread to prevent UI lag
frame = None
threading.Thread(target=capture_frame, daemon=True).start()

if run:
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        au_functions.add_gif(1)
        au_functions.instruction(1, lang)
        frame_count = 0

        while count <= 12:
            black = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image for pose landmarks
            frame_count += 1

            if frame is not None and frame_count % 5 == 0:  # Process every 5th frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw landmarks on black background
                mp_drawing.draw_landmarks(black, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                l = []
                if results.pose_landmarks:
                    a, b, c = results.pose_landmarks.landmark[0].x, results.pose_landmarks.landmark[0].y, results.pose_landmarks.landmark[0].z
                    landmarks = results.pose_landmarks.landmark
                    for i in landmarks:
                        l.extend([i.x - a, i.y - b, i.z - c])

                    l = np.array([l])
                    prediction = model.predict(l)
                    pos = np.argmax(prediction)
                    confidence = prediction[0][pos]

                    if curr_pos > 8:
                        curr_pos = 11 - curr_pos + 2

                    # Display Pose & Confidence
                    cv2.putText(image, f'Target Pose: {poses[curr_pos]}', (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f'Current Pose: {poses[pos]}', (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f'Confidence: {confidence:.2f}', (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f'Pose Count: {count}', (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

                    # Update Streamlit UI
                    accuracy_widget.metric("Prediction Accuracy", f"{confidence * 100:.2f}%")

                    if pos == curr_pos:
                        # Call appropriate function for pose correction
                        pose_functions = {
                            1: au_functions.Pranamasana, 2: au_functions.Hastauttanasana, 3: au_functions.Hastapadasana,
                            4: au_functions.Ashwa_Sanchalanasana_left, 5: au_functions.stick_pose,
                            6: au_functions.ashtanga, 7: au_functions.cobra, 8: au_functions.adho_mukha
                        }
                        if curr_pos in pose_functions:
                            status = pose_functions[curr_pos](landmarks, lang)

                        if status == 'perfect':
                            play_audio(r"C:\Users\karth\Desktop\TrainWithAI\duolingo_correct.mp3")
                            count += 1
                            curr_pos = count
                            au_functions.instruction(curr_pos, lang)
                            au_functions.add_gif(curr_pos)

                FRAME_WINDOW.image(image, channels="BGR", use_column_width=False)
                # FRAME_WINDOW.image(black, channels="BGR")

        cap.release()
        cv2.destroyAllWindows()

    FRAME_WINDOW.empty()
    play_audio(r"C:\Users\karth\Desktop\TrainWithAI\Bye Bye Bye - Deadpool _ English Song.mp3")
    st.balloons()
    au_functions.add_success_gif("success3.gif")
    st.success("Congratulations! You have successfully completed all the poses.")
