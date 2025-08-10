import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
from collections import deque
import time
import threading

with open("gesture_model.pkl", "rb") as f:
    static_model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

engine = pyttsx3.init()
voices = engine.getProperty('voices')
print("Available voices:")
for i, voice in enumerate(voices):
    print(f"{i}: {voice.name} - {voice.id}")
engine.setProperty('voice', voices[0].id)  # Change if no sound
engine.setProperty('rate', 150)

engine_lock = threading.Lock()

def speak(text):
    with engine_lock:
        engine.stop()
        print(f"Speaking out loud: {text}")
        engine.say(text)
        engine.runAndWait()
        time.sleep(0.2)  # pause to avoid overlap

print("Testing TTS...")
speak("This is a test. Can you hear me?")
print("TTS test finished.")

def extract_features(landmarks):
    wrist = landmarks[mp_hands.HandLandmark.WRIST.value]
    tips_ids = [mp_hands.HandLandmark.THUMB_TIP.value,
                mp_hands.HandLandmark.INDEX_FINGER_TIP.value,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value,
                mp_hands.HandLandmark.RING_FINGER_TIP.value,
                mp_hands.HandLandmark.PINKY_TIP.value]
    features = []
    for tip_id in tips_ids:
        tip = landmarks[tip_id]
        dist = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2 + (tip.z - wrist.z)**2)**0.5
        features.append(dist)
    return features

cap = cv2.VideoCapture(0)
wrist_x = deque(maxlen=15)
wrist_y = deque(maxlen=15)
last_gesture = None
stable_count = 0
min_frames = 5
spoken_gesture = None

with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = None

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            features = extract_features(lm.landmark)
            gesture = static_model.predict([features])[0]

            wrist_x.append(lm.landmark[mp_hands.HandLandmark.WRIST].x)
            wrist_y.append(lm.landmark[mp_hands.HandLandmark.WRIST].y)

            if len(wrist_x) == wrist_x.maxlen:
                dx = max(wrist_x) - min(wrist_x)
                dy_start = wrist_y[0]
                if dy_start < 0.3 and dx > 0.2:
                    gesture = "Hello"

        else:
            # Reset when no hand detected
            stable_count = 0
            last_gesture = None
            spoken_gesture = None

        if gesture == last_gesture:
            stable_count += 1
        else:
            stable_count = 1
            last_gesture = gesture

        if stable_count >= min_frames and gesture and gesture != spoken_gesture:
            print("Speaking:", gesture)
            speak(gesture)
            spoken_gesture = gesture

        cv2.putText(frame, gesture if gesture else " ", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Signaloud Motion-Aware", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()