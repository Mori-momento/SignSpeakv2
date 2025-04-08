import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from utils import normalize_landmarks

# Directory to save data
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

MASTER_CSV = os.path.join(DATA_DIR, 'landmark_data.csv')
if not os.path.exists(MASTER_CSV):
    columns = ['label'] + [f"lm_{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']]
    df_master = pd.DataFrame(columns=columns)
    df_master.to_csv(MASTER_CSV, index=False)

# ASL alphabet excluding 'j' and 'z'
ASL_LABELS = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y'
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("ASL Data Collection")
print("Excluded letters: j, z")
print("Press 'q' to quit at any time.\n")

for label in ASL_LABELS:
    save_path = os.path.join(DATA_DIR, f"{label.upper()}.csv")
    if not os.path.exists(save_path):
        # Create new CSV with header
        columns = ['label'] + [f"lm_{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']]
        df = pd.DataFrame(columns=columns)
        df.to_csv(save_path, index=False)

    print(f"\nStarting automatic recording for '{label.upper()}'...")

    saved_count = 0

    while saved_count < 300:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_world_landmarks:
            hand_landmarks = results.multi_hand_world_landmarks[0]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            norm_landmarks = normalize_landmarks(landmarks)

            # Save sample
            row = [label.upper()] + norm_landmarks.tolist()

            # Save to per-label CSV
            df = pd.read_csv(save_path)
            df.loc[len(df)] = row
            df.to_csv(save_path, index=False)

            # Save to master CSV
            df_master = pd.read_csv(MASTER_CSV)
            df_master.loc[len(df_master)] = row
            df_master.to_csv(MASTER_CSV, index=False)

            saved_count += 1
            print(f"Saved sample {saved_count}/300 for '{label.upper()}'")

        # Draw landmarks for feedback
        frame.flags.writeable = True
        if results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Label: {label.upper()}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('ASL Data Collection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("Exiting data collection.")
            exit()

    print(f"Collected 300 samples for '{label.upper()}'. Press Enter to continue to next label.")
    input()

cap.release()
cv2.destroyAllWindows()