import cv2
import mediapipe as mp
import pickle
from sklearn.ensemble import RandomForestClassifier

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DATA, LABELS = [], []

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

def capture_letter(letter, samples=50):
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        collected = 0
        while collected < samples:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                    features = extract_features(lm.landmark)
                    DATA.append(features)
                    LABELS.append(letter)
                    collected += 1
                    print(f"Captured {collected}/{samples} for letter {letter}")

            cv2.imshow("Capture Letter", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

for letter in LETTERS:
    input(f"Press Enter to capture letter '{letter}'...")
    capture_letter(letter)

clf = RandomForestClassifier()
clf.fit(DATA, LABELS)

with open("gesture_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Letter model saved as gesture_model.pkl")
