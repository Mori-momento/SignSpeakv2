from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load models
with open('asl_svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

cnn_model = tf.keras.models.load_model('asl_cnn_model.h5')

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize webcam

# Store latest normalized landmarks for on-demand prediction
latest_norm_landmarks = None
cap = cv2.VideoCapture(0)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Global state for predictions
latest_preds = {
    "svm": "",
    "cnn": "",
    "ensemble": "",
    "hand_detected": False
}

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    landmarks -= wrist
    max_value = np.max(np.abs(landmarks))
    if max_value > 0:
        landmarks /= max_value
    return landmarks.flatten()

def generate_frames():
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            latest_preds["hand_detected"] = True
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            norm_landmarks = normalize_landmarks(landmark_array).reshape(1, -1)
            global latest_norm_landmarks
            latest_norm_landmarks = norm_landmarks

            if frame_count % 5 == 0:
                try:
                    svm_probs = svm_model.predict_proba(norm_landmarks)[0]
                    svm_idx = np.argmax(svm_probs)
                    svm_label = label_encoder.inverse_transform([svm_idx])[0]

                    cnn_probs = cnn_model.predict(norm_landmarks, verbose=0)[0]
                    cnn_idx = np.argmax(cnn_probs)
                    cnn_label = label_encoder.inverse_transform([cnn_idx])[0]

                    avg_probs = (svm_probs + cnn_probs) / 2
                    ensemble_idx = np.argmax(avg_probs)
                    ensemble_label = label_encoder.inverse_transform([ensemble_idx])[0]

                    latest_preds["svm"] = svm_label
                    latest_preds["cnn"] = cnn_label
                    latest_preds["ensemble"] = ensemble_label
                except Exception as e:
                    print(f"Prediction error: {e}")
                    latest_preds["svm"] = latest_preds["cnn"] = latest_preds["ensemble"] = "Error"
        else:
            latest_preds["hand_detected"] = False
            latest_preds["svm"] = latest_preds["cnn"] = latest_preds["ensemble"] = ""

        # (Removed) Overlay ensemble prediction

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        frame_count += 1

@app.route('/')
def index():
    return render_template('index.html')
# Helper function for on-demand prediction
def predict_from_landmarks(norm_landmarks, mode):
    result = {}
    if norm_landmarks is None:
        result["hand_detected"] = False
        result["prediction"] = ""
        return result

    try:
        if mode == "svm":
            svm_probs = svm_model.predict_proba(norm_landmarks)[0]
            svm_idx = np.argmax(svm_probs)
            svm_label = label_encoder.inverse_transform([svm_idx])[0]
            result["hand_detected"] = True
            result["prediction"] = svm_label
            result["confidence"] = float(np.max(svm_probs))
        elif mode == "cnn":
            cnn_probs = cnn_model.predict(norm_landmarks, verbose=0)[0]
            cnn_idx = np.argmax(cnn_probs)
            cnn_label = label_encoder.inverse_transform([cnn_idx])[0]
            result["hand_detected"] = True
            result["prediction"] = cnn_label
            result["confidence"] = float(np.max(cnn_probs))
        elif mode == "ensemble":
            svm_probs = svm_model.predict_proba(norm_landmarks)[0]
            cnn_probs = cnn_model.predict(norm_landmarks, verbose=0)[0]
            avg_probs = (svm_probs + cnn_probs) / 2
            ensemble_idx = np.argmax(avg_probs)
            ensemble_label = label_encoder.inverse_transform([ensemble_idx])[0]
            result["hand_detected"] = True
            result["prediction"] = ensemble_label
            result["svm_confidence"] = float(np.max(svm_probs))
            result["cnn_confidence"] = float(np.max(cnn_probs))
        else:
            result["hand_detected"] = False
            result["prediction"] = ""
    except Exception as e:
        result["hand_detected"] = False
        result["prediction"] = "Error"
        result["error"] = str(e)
    return result

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predictions')

@app.route('/predictions', methods=["GET"])
def predictions():
    mode = request.args.get("mode", "ensemble")
    # Use latest_norm_landmarks for on-demand prediction
    result = predict_from_landmarks(latest_norm_landmarks, mode)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)