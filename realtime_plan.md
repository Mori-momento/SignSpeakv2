# Project Plan: Real-time ASL Recognition with Live Video

**Project Goal:** Create a web application that captures live video, detects the right hand, extracts, normalizes, and sends its 3D landmarks in real-time to a backend for prediction using SVM, CNN, and an ensemble model, displaying all three results dynamically alongside the video feed.

**Core Components & Changes:**

1.  **Frontend (`templates/index.html`):**
    *   **Input:** Replace the file upload form with a live `<video>` element for the webcam feed and a `<canvas>` element positioned directly over the video for drawing landmarks.
    *   **Landmark Detection (JavaScript):**
        *   Integrate the MediaPipe Hands JavaScript library (likely via CDN).
        *   Use `navigator.mediaDevices.getUserMedia` to access the webcam and stream it to the `<video>` element.
        *   Initialize MediaPipe Hands to process the video stream, configured to detect at least one hand (`maxNumHands: 1` or `2`).
        *   Implement the `onResults` callback function provided by MediaPipe.
    *   **Right Hand Filtering & Normalization (JavaScript):**
        *   Inside `onResults`, check `results.multiHandedness` to identify if the detected hand is the 'Right' hand. If `maxNumHands` is 1, we might assume it's the target hand, or adjust logic if needed.
        *   If the right hand is found, extract its 3D *world* landmarks (`results.multiHandWorldLandmarks`).
        *   **Crucially, implement the landmark normalization logic (translation relative to wrist, scaling based on hand size) directly in JavaScript.** This must precisely mirror the Python `normalize_landmarks` function to ensure consistency. This avoids sending raw landmarks and reduces backend load.
        *   Flatten the 21x3 normalized landmarks into a 1D array of 63 values.
    *   **Backend Communication (JavaScript):**
        *   Implement a throttling mechanism (e.g., using `setTimeout` or `requestAnimationFrame` with a timer) to send the normalized, flattened landmarks to the backend only at regular intervals (e.g., every 200-300 milliseconds). This prevents flooding the server and keeps the UI responsive.
        *   Use the `fetch` API to send a POST request containing the JSON payload `{"landmarks": [normalized_landmark_array]}` to a *new* backend endpoint (e.g., `/predict_realtime`).
    *   **Display (HTML/JavaScript):**
        *   Use MediaPipe's `drawingUtils` to draw the *image* landmarks (`results.multiHandLandmarks`) onto the `<canvas>` overlay for visual feedback.
        *   Create distinct `div` elements on the page to display the predictions received from the backend for SVM, CNN, and the Ensemble model, including their confidence scores.
        *   Update these `div`s dynamically when a response is received from the backend `fetch` call. Clear them if no hand is detected.

2.  **Backend (`app.py`):**
    *   **Endpoint Modification:**
        *   The existing root route (`/`) will now simply serve the modified `index.html` template. Remove the image processing logic from it.
        *   Create a *new* Flask route, e.g., `@app.route('/predict_realtime', methods=['POST'])`.
    *   **Prediction Logic:**
        *   This new endpoint will expect a JSON POST request containing the *already normalized and flattened* 63 landmark features from the frontend.
        *   It will parse the incoming JSON data.
        *   Reshape the 1D array into the `(1, 63)` shape required by the models.
        *   Feed this reshaped data into the pre-loaded SVM model (`svm_model.predict_proba`) and CNN model (`cnn_model.predict`).
        *   Calculate the ensemble prediction (e.g., by averaging the probability distributions from SVM and CNN).
        *   Use the loaded `label_encoder` to convert the predicted numerical labels (for SVM, CNN, and Ensemble) back into their string representations (e.g., 'A', 'B', 'Ok').
        *   Extract the confidence scores for the predicted class for all three results.
    *   **Response:**
        *   Return a JSON response containing the predictions and confidences for all three models, e.g., `{"svm": {"prediction": "A", "confidence": 0.95}, "cnn": {"prediction": "A", "confidence": 0.98}, "ensemble": {"prediction": "A", "confidence": 0.965}}`.
    *   **Model Loading:** Ensure the SVM model, CNN model, and LabelEncoder are still loaded efficiently when the Flask application starts.

3.  **Normalization Consistency:**
    *   The JavaScript normalization function *must* be an exact functional replica of the Python `normalize_landmarks` function found in `utils.py`. Any discrepancy will lead to incorrect predictions.

**High-Level Flow Diagram:**

```mermaid
graph LR
    subgraph Browser (Client-Side)
        A[Webcam Video] --> B{MediaPipe JS};
        B -- Raw Landmarks --> C{JS: Filter Right Hand};
        C -- Right Hand World Landmarks --> D{JS: Normalize Landmarks};
        D -- Normalized Landmarks --> E[JS: Throttle & Send];
        B -- Image Landmarks --> F[Canvas Overlay];
        E -- POST /predict_realtime --> G[Backend];
        G -- JSON Predictions --> H{JS: Update Display};
        H --> I[HTML Prediction DIVs];
    end

    subgraph Server (Flask Backend)
        G -- Receives Normalized Landmarks --> J{Predict Endpoint};
        J --> K[SVM Model];
        J --> L[CNN Model];
        K --> M{Ensemble Logic};
        L --> M;
        M -- All Predictions --> J;
        J --> G;
    end

    style F fill:#eee,stroke:#333,stroke-width:1px;
    style I fill:#eee,stroke:#333,stroke-width:1px;
```

**Potential Challenges & Mitigation:**

*   **Normalization Mismatch:** Ensure JS and Python normalization are identical. (Mitigation: Careful implementation and testing).
*   **Performance/Latency:** Real-time processing can be demanding. (Mitigation: Client-side normalization, throttling requests, potentially simplifying the CNN model if needed, efficient JS).
*   **Handedness Detection:** Reliably identifying the 'Right' hand might require `maxNumHands: 2` and checking `results.multiHandedness`. (Mitigation: Implement check based on MediaPipe output).
*   **Browser Compatibility:** `getUserMedia` and WebGL (for MediaPipe) require modern browsers. (Mitigation: Inform user of requirements).

**Success Criteria:**

*   The application displays a live webcam feed.
*   Landmarks are visibly overlaid on the detected right hand in real-time.
*   Predictions for SVM, CNN, and Ensemble models are displayed dynamically below/beside the video feed.
*   Predictions update smoothly as the user performs different signs with their right hand.
*   The application remains responsive without significant lag.