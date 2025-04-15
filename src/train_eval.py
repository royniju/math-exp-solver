### train_eval.py
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

from crnn_ctc_model import build_crnn_ctc_model
from data_loader import load_ctc_data, get_characters

# Constants
IMAGE_DIR = "dataset/images"
LABEL_CSV = "dataset/labels.csv"
IMG_WIDTH = 128
IMG_HEIGHT = 32
MAX_LABEL_LEN = 8
BATCH_SIZE = 16
EPOCHS = 50

# Load characters and data
characters = get_characters(LABEL_CSV)
num_classes = len(characters)
X, y, input_lengths, label_lengths = load_ctc_data(IMAGE_DIR, LABEL_CSV, IMG_WIDTH, IMG_HEIGHT, characters, MAX_LABEL_LEN)

# Train-validation split
X_train, X_val, y_train, y_val, len_train, len_val, label_len_train, label_len_val = train_test_split(
    X, y, input_lengths, label_lengths, test_size=0.2, random_state=42
)

# Build model
model, prediction_model = build_crnn_ctc_model(IMG_WIDTH, IMG_HEIGHT, num_classes, MAX_LABEL_LEN)
model.compile(optimizer=Adam(), loss={'ctc': lambda y_true, y_pred: y_pred})

# Train
model.fit(
    x={
        'input_image': X_train,
        'label': y_train,
        'input_length': len_train,
        'label_length': label_len_train
    },
    y=np.zeros(len(X_train)),  # dummy y for CTC loss
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(
        {
            'input_image': X_val,
            'label': y_val,
            'input_length': len_val,
            'label_length': label_len_val
        },
        np.zeros(len(X_val))
    )
)

# Save model
prediction_model.save("crnn_ctc_model.h5")

# Save characters for decoding later
import pickle
with open("characters.pkl", "wb") as f:
    pickle.dump(characters, f)

print("Training complete. Model and character list saved.")
