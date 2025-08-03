import cv2
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hides all TF warnings

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from deepface import DeepFace

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Error: Could not access webcam.")
    exit()
print("Webcam accessed successfully.")

# Frame and emotion setup
frame_count = 0
emotion = "Detecting..."
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # for DeepFace

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Clip to frame boundaries (optional safety check)
        x, y = max(x, 0), max(y, 0)
        face_roi = rgb_frame[y:y + h, x:x + w]

        if frame_count % 5 == 0:
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
            except Exception as e:
                print(" Emotion detection failed:", e)
                emotion = "Unknown"

        # Draw results
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # FPS display
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Real-time Emotion Detection", frame)
    frame_count += 1

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
