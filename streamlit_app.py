import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load models
mask_model = load_model("mask_model.h5")
emotion_model = load_model("emotion_model.h5")

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

st.title("😷 Face Mask & Emotion Detection (Live Camera)")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 6)

    for (x, y, w, h) in faces:
        face_color = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]

        # MASK
        face_mask = cv2.resize(face_color, (100, 100))
        face_mask = face_mask / 255.0
        face_mask = np.reshape(face_mask, (1, 100, 100, 3))

        mask_pred = mask_model.predict(face_mask, verbose=0)
        mask_label = "Mask" if np.argmax(mask_pred) == 0 else "No Mask"

        # EMOTION
        face_emotion = cv2.resize(face_gray, (48, 48))
        face_emotion = face_emotion / 255.0
        face_emotion = np.reshape(face_emotion, (1, 48, 48, 1))

        emotion_pred = emotion_model.predict(face_emotion, verbose=0)
        emotion_label = emotion_labels[np.argmax(emotion_pred)]

        label = f"{emotion_label} + {mask_label}"

        color = (0,255,0) if mask_label=="Mask" else (0,0,255)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    FRAME_WINDOW.image(frame, channels="BGR")

camera.release()