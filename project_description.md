# Project Requirements: Real-time ASL Sign Recognition using 3D Landmarks

**Version:** 1.0
**Date:** 2025-04-08
**Author:** Gemini

## 1. Introduction & Goal

This document outlines the technical requirements for building a system capable of recognizing a predefined set of static American Sign Language (ASL) hand signs in real-time. The system will utilize 3D hand landmark data extracted from a standard webcam feed, process this data through a machine learning model, and display the predicted sign via a web interface. The primary goal is to create a functional prototype demonstrating the feasibility of landmark-only ASL sign recognition for static poses.

## 2. Scope

### 2.1. In Scope

* **Sign Set:** Recognition of a fixed, predefined set of **static** ASL signs (e.g., 24 signs corresponding to letters or simple static words like 'Ok', 'Point'). The specific signs need to be defined during the data collection phase.
* **Input:** Single hand detection and landmark extraction from a live webcam feed.
* **Technology:** Use MediaPipe for hand landmark detection (both Python and JavaScript versions).
* **Data:** Utilize 3D world landmarks (`multi_hand_world_landmarks`) provided by MediaPipe.
* **Processing:** Implement data normalization specific to hand landmarks (translation and scaling).
* **Models:** Train two different Machine Learning classifiers using the normalized 3D landmark data:
  * A Support Vector Machine (SVM)
  * A Convolutional Neural Network (CNN)
* **Application:** Develop a Flask-based web application for real-time prediction.
* **Interface:** A simple web frontend (HTML/JavaScript) displaying the webcam feed, detected landmarks (optional), and the predicted ASL sign with a confidence score.
* **Environment:** Local development and execution environment.

### 2.2. Out of Scope

* **Dynamic Signs:** Recognition of signs requiring motion analysis (e.g., 'J', 'Z', signs involving hand movement paths).
* **Full ASL Vocabulary:** Recognition beyond the predefined static sign set.
* **Two-Handed Signs:** Recognition requiring the analysis of both hands simultaneously.
* **Fingerspelling (complex):** Recognition of the complete fingerspelling process, especially letters involving motion.
* **Sentence Translation:** Converting sequences of signs into spoken/written language.
* **Deployment:** Cloud deployment, containerization (e.g., Docker), or packaging for distribution.
* **User Management:** User accounts, authentication, or saving user-specific data.
* **Advanced UI/UX:** Complex user interface features beyond basic display and feedback.
* **Robustness to Extreme Conditions:** Guaranteed performance under highly variable lighting, occlusions, or non-standard camera angles.

## 3. System Architecture

The system comprises the following core modules:

1.  **Data Collection Module (`data_collection.py`):**
    * Accesses the webcam using OpenCV.
    * Uses MediaPipe (Python) to detect hands and extract 3D world landmarks.
    * Applies normalization to the landmarks.
    * Saves normalized landmarks along with user-provided labels to a persistent format (CSV).
2.  **Data Normalization Module (Function):**
    * A reusable function (defined in Python for collection/training/backend, conceptually similar logic in JS if needed) that performs:
        * Translation: Makes landmarks relative to the wrist (landmark 0).
        * Scaling: Normalizes landmark coordinates based on a consistent hand measurement (e.g., distance between wrist and middle finger MCP joint - landmarks 0 & 9).
    * **Crucially, this logic must be identical wherever applied (collection, training, inference).**
3.  **Model Training Module (`train_model.py`):**
    * Loads the collected/normalized landmark data (from CSV).
    * Encodes string labels into numerical format.
    * Splits data into training and testing sets.
    * Trains two classifiers:
        * An SVM using `scikit-learn`.
        * A CNN using TensorFlow/Keras.
    * Evaluates the performance of both models.
    * Saves the trained SVM model (`.pkl`), the CNN model (`.h5` or SavedModel format), and the label encoder (`.pkl`).
4.  **Real-time Prediction API (Flask Backend - `app.py`):**
    * Loads the pre-trained SVM model and label encoder.
    * Provides an HTTP endpoint (`/predict`) to receive landmark data (JSON) from the frontend.
    * Applies the **identical** normalization logic to the received landmarks.
    * Feeds the normalized data into the loaded model for prediction.
    * Returns the predicted sign label and confidence score (JSON).
    * Provides an endpoint (`/`) to serve the main web page.
5.  **Web Frontend (`templates/index.html`):**
    * Uses HTML, CSS, and JavaScript.
    * Accesses the user's webcam via `navigator.mediaDevices.getUserMedia`.
    * Uses MediaPipe (JavaScript) to detect hands and extract 3D world landmarks from the video stream.
    * Sends the extracted landmarks (flattened) to the Flask backend's `/predict` endpoint using the `Workspace` API.
    * Receives the prediction result and displays it dynamically on the page.
    * Optionally visualizes the detected landmarks on a canvas overlaying the video feed.

## 4. Data Requirements

* **Data Source:** Live video stream from a standard USB or built-in webcam.
* **Raw Data Type:** 3D Hand World Landmarks (21 landmarks per hand, each with x, y, z coordinates) provided by `results.multi_hand_world_landmarks` in MediaPipe. These coordinates are relative to the hand's approximate geometric center and scaled roughly in metric units.
* **Collected Data Format:** CSV file (`landmark_data.csv`).
    * **Columns:** `label` (string), `lm_0_x`, `lm_0_y`, `lm_0_z`, `lm_1_x`, ..., `lm_20_z` (63 feature columns total).
* **Normalization Procedure:**
    1.  Represent landmarks as a (21, 3) NumPy array.
    2.  Calculate translation vector: Coordinates of the wrist (landmark 0).
    3.  Translate all landmarks by subtracting the translation vector.
    4.  Calculate scale factor: Euclidean distance between the (translated) wrist (now at origin) and the middle finger MCP joint (landmark 9). Handle potential division by zero if distance is negligible.
    5.  Scale all translated landmarks by dividing by the scale factor.
    6.  Flatten the normalized (21, 3) array into a 1D vector of 63 features.
* **Labels:** A defined set of static ASL sign labels (e.g., 'A', 'B', 'C', 'Ok', 'Point', 'Five', ...). The exact list must be finalized before data collection.
* **Quantity:** Minimum of 200-300 samples per sign label recommended for reasonable initial performance. More data generally leads to better robustness.
* **Diversity:** Data should ideally be collected under varying (but reasonable) lighting conditions, slightly different camera angles/distances, and potentially from different individuals to improve model generalization.

## 5. Machine Learning Model

### 5.1 Support Vector Machine (SVM) Classifier
* **Implementation:** `sklearn.svm.SVC`.
* **Kernel:** Radial Basis Function (RBF) (`kernel='rbf'`).
* **Parameters:** Use default parameters initially (`C=1.0`, `gamma='scale'`). Enable probability estimates (`probability=True`) for confidence scores. Use `random_state` for reproducibility.
* **Input Features:** A single vector of 63 normalized landmark coordinates (output of the normalization step).
* **Training Data:** The `landmark_data.csv` file generated by the collection script.
* **Data Split:** 80% Training, 20% Testing (or similar standard split). Use stratified splitting (`stratify=y`) to maintain class proportions.
* **Label Handling:** Use `sklearn.preprocessing.LabelEncoder` to convert string labels to integers for training and back for interpretation. The fitted encoder must be saved.
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score (per class), Confusion Matrix. Calculated on the test set.
* **Saved Artifacts:**
    * Trained SVM model object (saved via `pickle` as `asl_svm_model.pkl`).

### 5.2 Convolutional Neural Network (CNN) Classifier
* **Implementation:** TensorFlow/Keras.
* **Input Features:** The same normalized landmark data, reshaped as a (21, 3, 1) tensor or flattened vector input.
* **Architecture:** A lightweight CNN or MLP suitable for small input size, e.g., a few Conv1D or Dense layers with ReLU activations and dropout for regularization.
* **Training Data:** Same as SVM.
* **Data Split:** Same as SVM.
* **Label Handling:** Same label encoding as SVM; reuse the saved encoder.
* **Training Parameters:** Use categorical cross-entropy loss, Adam optimizer, early stopping based on validation loss.
* **Evaluation Metrics:** Same as SVM.
* **Saved Artifacts:**
    * Trained CNN model saved in `.h5` format or TensorFlow SavedModel (`asl_cnn_model.h5` or directory).
    * (Reuse) Fitted LabelEncoder object (`label_encoder.pkl`).

## 6. Software Components & Technologies

* **Programming Language:** Python (version 3.8 or higher recommended)
* **Python Libraries (`requirements.txt`):**
    * `Flask`: Web server framework.
    * `opencv-python`: Webcam interaction, image handling (primarily for data collection).
    * `mediapipe`: Hand tracking and landmark extraction (Python version).
    * `numpy`: Numerical computation (landmark manipulation, array handling).
    * `scikit-learn`: SVM implementation, data splitting, label encoding, evaluation metrics.
    * `pandas`: Reading and handling the CSV data file.
    * `pickle`: Saving and loading Python objects (model, encoder).
* **Frontend Technologies:**
    * HTML5: Structure of the web page.
    * CSS3: Basic styling for layout and appearance.
    * JavaScript (ES6+):
        * `navigator.mediaDevices.getUserMedia`: Accessing webcam feed.
        * MediaPipe Hands for Web (via CDN): Landmark extraction in the browser.
        * `Workspace` API: Asynchronous communication with the Flask backend.
        * DOM manipulation: Displaying predictions and video feed.
        * Canvas API: Drawing landmark visualizations (optional).

## 7. Application Details (Flask & Frontend)

### 7.1. Backend (`app.py`)

* **Initialization:** Load the `asl_svm_model.pkl` and `label_encoder.pkl` upon starting the Flask application. Handle file not found errors gracefully.
* **Root Endpoint (`@app.route('/')`):**
    * Method: GET
    * Action: Renders and returns the `templates/index.html` file.
* **Prediction Endpoint (`@app.route('/predict')`):**
    * Method: POST
    * Request Body: JSON object `{"landmarks": [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]}` (a flat list/array of 63 floating-point numbers).
    * Action:
        1.  Validate the incoming JSON payload.
        2.  Apply the **exact same** `normalize_landmarks_inference` function to the received `landmarks` list.
        3.  Reshape the normalized vector into the `[1, 63]` format expected by scikit-learn's `predict` method.
        4.  Call `model.predict()` and `model.predict_proba()` on the prepared feature vector.
        5.  Use the loaded `label_encoder` to convert the predicted numerical label back to its string representation.
        6.  Extract the confidence score for the predicted class from the probabilities.
    * Response Body (Success): JSON object `{"prediction": "SIGN_LABEL", "confidence": 0.XX}`.
    * Response Body (Error): JSON object `{"error": "Error description"}` with appropriate HTTP status code (e.g., 400 for bad input, 500 for server error).
* **Normalization:** Include the landmark normalization function directly within `app.py` or import it from a shared utility module, ensuring it's identical to the one used in `data_collection.py`.

### 7.2. Frontend (`templates/index.html`)

* **Structure:** Basic HTML layout with a `<video>` element for the webcam feed, a `<canvas>` element positioned over the video for drawing, and a `<div>` area to display predictions.
* **Webcam Access:** Use `navigator.mediaDevices.getUserMedia` to request access and stream video to the `<video>` element.
* **MediaPipe Integration:**
    * Load the MediaPipe Hands solution (e.g., from CDN).
    * Initialize Hands, setting appropriate options (`maxNumHands: 1`, `modelComplexity: 1`, etc.).
    * Set up the `onResults` callback function.
* **`onResults` Callback Logic:**
    1.  Check if `results.multiHandWorldLandmarks` contains data for at least one hand.
    2.  If landmarks exist:
        * Extract the landmarks for the first detected hand.
        * Flatten the 21x3 landmarks into a single array of 63 elements.
        * Optionally, use `drawing_utils` (from MediaPipe JS) to draw the *image* landmarks (`results.multiHandLandmarks`) onto the `<canvas>` for visualization.
        * Periodically (e.g., using a timer or throttling mechanism like every 200ms) send the **flattened world landmarks** to the `/predict` endpoint via a `Workspace` POST request with a JSON body.
    3.  Handle the `Workspace` response: Parse the JSON, update the prediction display `<div>` with the `prediction` and `confidence`. Handle potential errors returned from the backend.
    4.  If no hand is detected, update the display accordingly (e.g., "No hand detected").

## 8. Non-Functional Requirements

* **Performance:** Predictions should appear near real-time. The primary latency will be from MediaPipe's landmark detection and network communication. Target perception of < 300-500ms delay between gesture and displayed prediction.
* **Usability:** The data collection script should provide clear command-line instructions. The web application should be intuitive, showing the video feed and the prediction clearly.
* **Maintainability:** Code should be reasonably commented. Separate Python scripts for distinct tasks (collection, training, app). The critical normalization logic should be centralized or meticulously duplicated.
* **Reliability:** The system should handle common error conditions gracefully: webcam not found, no hand detected, failure to load model files, invalid data received by the backend.

## 9. Setup and Execution

1.  **Clone Repository:** Obtain the project source code.
2.  **Install Dependencies:** `pip install -r requirements.txt` (A `requirements.txt` file should be created listing all Python libraries from section 6).
3.  **Collect Data:** Run `python data_collection.py`. Follow on-screen prompts to label and capture samples for each required ASL sign. Press 'q' to finish.
4.  **Train Model:** Run `python train_model.py`. This will process the collected data and save the model artifacts (`.pkl` files) in the `asl_model` directory. Review the evaluation metrics printed to the console.
5.  **Run Application:** Execute `python app.py`. Note the URL provided (e.g., `http://127.0.0.1:5000`).
6.  **Access Interface:** Open the URL from step 5 in a web browser that supports `getUserMedia` and WebGL (most modern browsers). Allow webcam access when prompted.
7.  **Test:** Perform the trained ASL signs in front of the webcam and observe the predictions.

## 10. Future Considerations (Optional Enhancements)

* **Dynamic Sign Support:** Integrate sequence models (LSTM, GRU, Transformers) using sequences of landmarks over time. Requires video data collection.
* **Two-Handed Sign Support:** Modify landmark extraction and model input to handle data from two hands.
* **Client-Side Prediction:** Convert the trained model (e.g., to TensorFlow.js or ONNX format) and perform normalization and prediction entirely within the browser using JavaScript for lower latency.
* **Improved Robustness:** Explore more advanced data augmentation techniques during training.
* **Deployment:** Containerize the application using Docker for easier deployment.
* **UI Improvements:** Enhance the user interface for better feedback and potentially configuration options.