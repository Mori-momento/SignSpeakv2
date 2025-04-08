from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import tensorflow as tf
import mediapipe as mp
import cv2
import os
# Assuming normalize_landmarks is primarily used client-side now,
# but keeping it here in case needed for other purposes or future server-side fallback.
from utils import normalize_landmarks

app = Flask(__name__)

# --- Model and Encoder Loading ---
try:
    with open('asl_svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    print("SVM model loaded successfully.")
except FileNotFoundError:
    print("Error: SVM model file 'asl_svm_model.pkl' not found.")
    svm_model = None
except Exception as e:
    print(f"Error loading SVM model: {e}")
    svm_model = None

try:
    # Ensure custom objects aren't needed or handle them if they are
    cnn_model = tf.keras.models.load_model('asl_cnn_model.h5', compile=False) # Added compile=False for faster loading if optimizer state isn't needed for inference
    print("CNN model loaded successfully.")
except FileNotFoundError:
    print("Error: CNN model file 'asl_cnn_model.h5' not found.")
    cnn_model = None
except Exception as e:
    print(f"Error loading CNN model: {e}")
    cnn_model = None

try:
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("Label encoder loaded successfully.")
except FileNotFoundError:
    print("Error: Label encoder file 'label_encoder.pkl' not found.")
    label_encoder = None
except Exception as e:
    print(f"Error loading label encoder: {e}")
    label_encoder = None

# --- Routes ---

@app.route('/')
def index():
    """Serves the main HTML page with webcam feed and prediction display."""
    return render_template('index.html')

@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
    """Receives normalized landmarks from frontend and returns predictions."""
    if not svm_model or not cnn_model or not label_encoder:
        return jsonify({"error": "Models or encoder not loaded properly on server."}), 500

    data = request.get_json()
    if not data or 'landmarks' not in data:
        return jsonify({"error": "Missing landmark data"}), 400

    try:
        # Landmarks are expected to be normalized and flattened by the client
        norm_landmarks_flat = np.array(data['landmarks'])

        if norm_landmarks_flat.shape[0] != 63:
             return jsonify({"error": f"Invalid landmark data shape: expected 63 features, got {norm_landmarks_flat.shape[0]}"}), 400

        # Reshape for model prediction
        norm_landmarks = norm_landmarks_flat.reshape(1, -1) # Shape (1, 63)

        # --- SVM Prediction ---
        svm_probs = svm_model.predict_proba(norm_landmarks)[0]
        svm_idx = np.argmax(svm_probs)
        svm_pred_label = label_encoder.inverse_transform([svm_idx])[0]
        svm_conf = float(svm_probs[svm_idx]) # Ensure JSON serializable

        # --- CNN Prediction ---
        cnn_probs = cnn_model.predict(norm_landmarks, verbose=0)[0]
        cnn_idx = np.argmax(cnn_probs)
        cnn_pred_label = label_encoder.inverse_transform([cnn_idx])[0]
        cnn_conf = float(cnn_probs[cnn_idx]) # Ensure JSON serializable

        # --- Ensemble Prediction (Average Probabilities) ---
        # Ensure probability arrays have the same length (number of classes)
        if len(svm_probs) != len(cnn_probs):
             return jsonify({"error": "Model output dimensions mismatch for ensemble."}), 500

        avg_probs = (svm_probs + cnn_probs) / 2
        ensemble_idx = np.argmax(avg_probs)
        ensemble_pred_label = label_encoder.inverse_transform([ensemble_idx])[0]
        ensemble_conf = float(avg_probs[ensemble_idx]) # Ensure JSON serializable

        return jsonify({
            "svm": {"prediction": svm_pred_label, "confidence": svm_conf},
            "cnn": {"prediction": cnn_pred_label, "confidence": cnn_conf},
            "ensemble": {"prediction": ensemble_pred_label, "confidence": ensemble_conf}
        })

    except Exception as e:
        print(f"Prediction error: {e}") # Log the error server-side
        return jsonify({"error": "Prediction failed on server."}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on the network if needed, otherwise 127.0.0.1
    app.run(host='0.0.0.0', port=5000, debug=True)