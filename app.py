from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import tensorflow as tf
import mediapipe as mp
import cv2
import os
from utils import normalize_landmarks

app = Flask(__name__)

# Load models and label encoder
with open('asl_svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

cnn_model = tf.keras.models.load_model('asl_cnn_model.h5')

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    svm_pred = None
    cnn_pred = None
    ensemble_pred = None
    svm_conf = None
    cnn_conf = None
    ensemble_conf = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            # Read image
            img_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract landmarks
            results = hands.process(img_rgb)
            if results.multi_hand_world_landmarks:
                hand_landmarks = results.multi_hand_world_landmarks[0]
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                norm_landmarks = normalize_landmarks(landmarks).reshape(1, -1)

                # SVM prediction
                svm_probs = svm_model.predict_proba(norm_landmarks)[0]
                svm_idx = np.argmax(svm_probs)
                svm_pred = label_encoder.inverse_transform([svm_idx])[0]
                svm_conf = svm_probs[svm_idx]

                # CNN prediction
                cnn_probs = cnn_model.predict(norm_landmarks, verbose=0)[0]
                cnn_idx = np.argmax(cnn_probs)
                cnn_pred = label_encoder.inverse_transform([cnn_idx])[0]
                cnn_conf = cnn_probs[cnn_idx]

                # Ensemble: average probabilities
                avg_probs = (svm_probs + cnn_probs) / 2
                ensemble_idx = np.argmax(avg_probs)
                ensemble_pred = label_encoder.inverse_transform([ensemble_idx])[0]
                ensemble_conf = avg_probs[ensemble_idx]

            else:
                prediction = "No hand detected."

    return render_template('index.html',
                           svm_pred=svm_pred, svm_conf=svm_conf,
                           cnn_pred=cnn_pred, cnn_conf=cnn_conf,
                           ensemble_pred=ensemble_pred, ensemble_conf=ensemble_conf)

if __name__ == '__main__':
    app.run(debug=True)