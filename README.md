# SignSpeak - American Sign Language Recognition

SignSpeak is a Python-based project designed to recognize American Sign Language (ASL) gestures using computer vision and machine learning techniques. It provides tools for data collection, model training, and a web interface for real-time ASL recognition.

---

## Features

- **Data Collection:** Capture hand landmark data for ASL gestures.
- **Model Training:** Train CNN and SVM models on collected data.
- **Real-Time Recognition:** Web app interface for live ASL gesture recognition.
- **Modular Design:** Separate modules for data handling, model training, and inference.

---

## Project Structure

```
.
├── app.py                   # Flask web application for real-time ASL recognition
├── data_collection.py       # Script to collect ASL gesture data
├── train_model.py           # Script to train machine learning models
├── utils.py                 # Utility functions
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── project_description.md   # Detailed project description
├── realtime_plan.md         # Real-time system design plan
├── asl_cnn_model.h5         # Trained CNN model
├── asl_svm_model.pkl        # Trained SVM model
├── landmark_data.csv        # Collected landmark data
├── data/                    # Directory for raw data
├── templates/
│   └── index.html           # Web app HTML template
└── .gitignore               # Git ignore rules
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository_url>
cd SignSpeak6
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Data Collection

Run the data collection script to capture ASL gesture landmarks:

```bash
python data_collection.py
```

### Model Training

Train the models using the collected data:

```bash
python train_model.py
```

### Run the Web Application

Start the Flask app for real-time ASL recognition:

```bash
python app.py
```

Then, open your browser and navigate to `http://127.0.0.1:5000` to use the app.

---

## Dependencies

All required Python packages are listed in `requirements.txt`. Key dependencies include:

- OpenCV
- MediaPipe
- TensorFlow
- scikit-learn
- Flask

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or collaboration, please contact the project maintainer.