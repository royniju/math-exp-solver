import os
import numpy as np
import tensorflow as tf
import cv2
import pickle

# Load saved model and characters
model = tf.keras.models.load_model("crnn_ctc_model.h5", compile=False)
with open("characters.pkl", "rb") as f:
    characters = pickle.load(f)

# Mapping index to character
idx_to_char = {i: ch for i, ch in enumerate(characters)}

def preprocess_image(image_path, img_width=128, img_height=32):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to (width, height)
    img = cv2.resize(img, (img_width, img_height))  # (128, 32)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Reshape to (batch, width, height, channels)
    img = np.expand_dims(img, axis=-1)           # (128, 32, 1)
    img = np.transpose(img, (1, 0, 2))            # (32, 128, 1)
    img = np.expand_dims(img, axis=0)            # (1, 32, 128, 1)

    return img

def decode_prediction(pred):
    decoded, _ = tf.keras.backend.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1])
    pred_indices = decoded[0].numpy()
    pred_text = ""
    for i in range(pred_indices.shape[1]):
        index = pred_indices[0][i]
        if index != -1:
            pred_text += idx_to_char.get(index, "")
    return pred_text

# ----------- USAGE EXAMPLE --------------
image_path = "C:/Users/nijuk/Documents/GitHub/math-exp-solver/dataset/images/img_428.png"

img = preprocess_image(image_path)
pred = model.predict(img)
decoded = decode_prediction(pred)

print(f"Predicted expression: {decoded}")
