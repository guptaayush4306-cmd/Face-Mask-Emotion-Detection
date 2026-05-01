import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load models
mask_model = load_model("mask_model.h5")
emotion_model = load_model("emotion_model.h5")

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())
print("Press ESC to exit")

# History for smoothing
emo_hist = []
mask_hist = []

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Camera not working!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improved face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face_color = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]

        # ===== MASK =====
        face_mask = cv2.resize(face_color, (100, 100))
        face_mask = face_mask / 255.0
        face_mask = np.reshape(face_mask, (1, 100, 100, 3))

        mask_pred = mask_model.predict(face_mask, verbose=0)[0]
        mask_idx = int(np.argmax(mask_pred))
        mask_conf = float(np.max(mask_pred))

        # ===== EMOTION =====
        face_emotion = cv2.resize(face_gray, (48, 48))
        face_emotion = face_emotion / 255.0
        face_emotion = np.reshape(face_emotion, (1, 48, 48, 1))

        emotion_pred = emotion_model.predict(face_emotion, verbose=0)[0]
        emo_idx = int(np.argmax(emotion_pred))
        emo_conf = float(np.max(emotion_pred))

        # ===== SMOOTHING =====
        emo_hist.append(emo_idx)
        mask_hist.append(mask_idx)

        N = 5
        emo_hist = emo_hist[-N:]
        mask_hist = mask_hist[-N:]

        emo_idx = max(set(emo_hist), key=emo_hist.count)
        mask_idx = max(set(mask_hist), key=mask_hist.count)

        emotion_label = emotion_labels[emo_idx]
        mask_label = "Mask" if mask_idx == 0 else "No Mask"

        # ===== DISPLAY =====
        label = f"{emotion_label} ({emo_conf:.2f}) + {mask_label} ({mask_conf:.2f})"

        color = (0, 255, 0) if mask_label == "Mask" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Mask + Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()