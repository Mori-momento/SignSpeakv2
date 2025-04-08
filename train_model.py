import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

from utils import normalize_landmarks

# Load data
df = pd.read_csv('data/landmark_data.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Train SVM
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Save SVM model
with open('asl_svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

# Build simple CNN/MLP
tf.random.set_seed(42)

model = models.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_cnn_model.h5', save_best_only=True)
]

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"CNN Test accuracy: {acc:.2f}")

y_pred_cnn = np.argmax(model.predict(X_test), axis=1)
print("CNN Classification Report:")
print(classification_report(y_test, y_pred_cnn))

# Save CNN model
model.save('asl_cnn_model.h5')

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)