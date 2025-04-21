# SignSpeak Project Requirements Document

**Version:** 1.0  
**Date:** 2023-07-15  
**Author:** Augment AI

## 1. Executive Summary

SignSpeak is a real-time American Sign Language (ASL) recognition application that uses computer vision and machine learning to interpret hand gestures. The project combines MediaPipe for hand landmark detection with custom-trained SVM and CNN models for gesture classification. The application features a modern, neumorphic design for an intuitive user experience, with real-time video processing and immediate visual feedback.

## 2. Project Objectives

### 2.1 Primary Objectives

- Create a web-based application that can recognize static American Sign Language (ASL) gestures in real-time
- Provide an intuitive, accessible user interface for both signers and learners
- Implement accurate machine learning models for ASL gesture recognition
- Enable real-time text transcription of recognized signs

### 2.2 Success Criteria

- Recognition accuracy of at least 85% for the supported static ASL signs
- Real-time processing with latency under 300ms
- Intuitive user interface with clear visual feedback
- Ability to save and download transcribed text
- Support for different video display modes to enhance visibility

## 3. Scope

### 3.1 In Scope

- **Sign Set:** Recognition of a fixed, predefined set of static ASL signs (24 letters excluding 'j' and 'z' which require motion)
- **Input:** Single hand detection and landmark extraction from a live webcam feed
- **Technology:** MediaPipe for hand landmark detection (both Python and JavaScript versions)
- **Data:** 3D world landmarks provided by MediaPipe
- **Processing:** Data normalization specific to hand landmarks (translation and scaling)
- **Models:** Two machine learning classifiers:
  - Support Vector Machine (SVM)
  - Convolutional Neural Network (CNN)
  - Ensemble approach combining both models
- **Application:** Flask-based web application for real-time prediction
- **Interface:** Web frontend displaying webcam feed, detected landmarks, and predicted ASL signs
- **Features:**
  - Real-time sign recognition
  - Text transcription
  - Video display modes (negative and black & white)
  - Downloadable transcript
  - ASL reference chart
  - Onboarding tutorial

### 3.2 Out of Scope

- **Dynamic Signs:** Recognition of signs requiring motion analysis (e.g., 'j', 'z')
- **Full ASL Vocabulary:** Recognition beyond the predefined static sign set
- **Two-Handed Signs:** Recognition requiring the analysis of both hands simultaneously
- **Fingerspelling (complex):** Recognition of the complete fingerspelling process
- **Sentence Translation:** Converting sequences of signs into grammatically correct spoken/written language
- **Cloud Deployment:** Containerization or cloud hosting setup
- **User Management:** User accounts, authentication, or saving user-specific data

## 4. System Architecture

### 4.1 High-Level Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Client Browser │◄────┤  Flask Server   │◄────┤  ML Models      │
│  (HTML/JS/CSS)  │     │  (Python)       │     │  (SVM & CNN)    │
│                 │─────►                 │─────►                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 4.2 Component Breakdown

1. **Data Collection Module (`data_collection.py`):**
   - Accesses webcam using OpenCV
   - Uses MediaPipe to detect hands and extract 3D world landmarks
   - Applies normalization to landmarks
   - Saves normalized landmarks with labels to CSV format

2. **Data Normalization Module (`utils.py`):**
   - Translates landmarks relative to the wrist (landmark 0)
   - Scales landmarks based on consistent hand measurement
   - Ensures identical normalization across collection, training, and inference

3. **Model Training Module (`train_model.py`):**
   - Loads collected landmark data
   - Encodes string labels into numerical format
   - Splits data into training and testing sets
   - Trains SVM and CNN classifiers
   - Evaluates model performance
   - Saves trained models and label encoder

4. **Real-time Prediction API (`app.py`):**
   - Loads pre-trained models and label encoder
   - Provides endpoints for serving the web application
   - Processes video frames and extracts hand landmarks
   - Performs real-time prediction using both models
   - Returns prediction results to the frontend

5. **Web Frontend (`templates/index.html`):**
   - Displays webcam feed with landmark overlay
   - Shows prediction results in real-time
   - Provides UI controls for video display modes
   - Enables text transcription and download
   - Includes onboarding tutorial and ASL reference chart

## 5. Functional Requirements

### 5.1 Data Collection

- System shall capture hand landmark data for ASL gestures
- System shall normalize landmark data for consistent processing
- System shall save data in a structured format (CSV)
- System shall collect multiple samples per gesture for training robustness

### 5.2 Model Training

- System shall train an SVM model on normalized landmark data
- System shall train a CNN model on normalized landmark data
- System shall evaluate model performance with appropriate metrics
- System shall save trained models for later use in inference

### 5.3 Real-time Recognition

- System shall access the user's webcam feed
- System shall detect and track hand landmarks in real-time
- System shall normalize detected landmarks
- System shall predict ASL signs using both SVM and CNN models
- System shall combine predictions using an ensemble approach
- System shall display prediction results with confidence scores

### 5.4 User Interface

- System shall display the webcam feed with hand landmark overlay
- System shall provide toggle for different video display modes (negative and black & white)
- System shall show prediction results in real-time
- System shall transcribe recognized signs into text
- System shall allow users to download the transcribed text
- System shall provide an ASL reference chart
- System shall include an onboarding tutorial for first-time users
- System shall support keyboard shortcuts for text manipulation (space for new line, backspace for deletion)

## 6. Non-Functional Requirements

### 6.1 Performance

- System shall process video frames at a minimum of 15 FPS
- System shall provide predictions with latency under 300ms
- System shall handle continuous operation without memory leaks

### 6.2 Usability

- Interface shall be intuitive and accessible
- System shall provide clear visual feedback for recognized signs
- System shall include helpful instructions for users
- System shall support responsive design for different screen sizes

### 6.3 Reliability

- System shall handle cases where no hand is detected
- System shall provide graceful error handling
- System shall maintain consistent performance over extended use

### 6.4 Compatibility

- System shall work on modern web browsers (Chrome, Firefox, Edge, Safari)
- System shall support both desktop and mobile devices with cameras
- System shall degrade gracefully on unsupported browsers

## 7. Technical Stack

### 7.1 Backend

- **Language:** Python 3.8+
- **Web Framework:** Flask
- **Computer Vision:** OpenCV, MediaPipe
- **Machine Learning:** TensorFlow, scikit-learn
- **Data Processing:** NumPy, Pandas

### 7.2 Frontend

- **Languages:** HTML5, CSS3, JavaScript (ES6+)
- **Libraries:** MediaPipe Hands (JavaScript)
- **Design:** Neumorphic UI components

### 7.3 Development Tools

- **Version Control:** Git
- **Environment:** Virtual environment (venv)
- **Testing:** Manual testing for UI, automated testing for models

## 8. Implementation Plan

### 8.1 Phase 1: Data Collection and Preprocessing

- Set up webcam access and MediaPipe integration
- Implement landmark normalization
- Create data collection script
- Collect and preprocess ASL gesture data

### 8.2 Phase 2: Model Training

- Implement SVM model training
- Implement CNN model training
- Evaluate and optimize models
- Save trained models

### 8.3 Phase 3: Backend Development

- Set up Flask application
- Implement real-time prediction endpoints
- Integrate trained models
- Optimize for performance

### 8.4 Phase 4: Frontend Development

- Create basic UI layout
- Implement webcam integration
- Add landmark visualization
- Develop prediction display
- Implement text transcription
- Add video display modes
- Create onboarding tutorial
- Design ASL reference chart

### 8.5 Phase 5: Testing and Refinement

- Test recognition accuracy
- Optimize performance
- Refine user interface
- Fix bugs and issues

## 9. Future Enhancements

- **Dynamic Sign Support:** Integrate sequence models for signs requiring motion
- **Two-Handed Sign Support:** Extend to recognize signs using both hands
- **Client-Side Prediction:** Convert models to TensorFlow.js for browser-based inference
- **Improved Robustness:** Implement advanced data augmentation techniques
- **Deployment:** Containerize application for easier deployment
- **UI Improvements:** Add dark mode, sound feedback, and gesture history

## 10. Appendix

### 10.1 Glossary

- **ASL:** American Sign Language
- **SVM:** Support Vector Machine
- **CNN:** Convolutional Neural Network
- **MediaPipe:** Google's open-source framework for building multimodal applied ML pipelines
- **Landmark:** A specific point on the hand (e.g., fingertip, knuckle) detected by MediaPipe
- **Neumorphic Design:** A UI design style that combines elements of skeuomorphism and flat design

### 10.2 References

- MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands.html
- TensorFlow: https://www.tensorflow.org/
- scikit-learn: https://scikit-learn.org/
- Flask: https://flask.palletsprojects.com/
