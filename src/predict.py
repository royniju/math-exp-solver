import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Load trained model
model = tf.keras.models.load_model("symbol_classifier_tf_model.keras")

# Rebuild class label mapping from directory
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '.', '/', '=', '*', '-']
#print("Class names:", class_names)

# --- Load and preprocess test image ---
img = cv2.imread("src/test2.png", cv2.IMREAD_GRAYSCALE)
_, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find character contours
contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bounding_boxes = [cv2.boundingRect(c) for c in contours]
bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])  # left to right

expression = ""

for (x, y, w, h) in bounding_boxes:
    char_img = img[y:y+h, x:x+w]

    # Add white padding to make the image square before resizing
    padding = 10
    size = max(w, h) + 2 * padding
    square_img = np.ones((size, size), dtype=np.uint8) * 255  # white background
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square_img[y_offset:y_offset+h, x_offset:x_offset+w] = char_img

    # Resize and normalize
    resized_img = cv2.resize(square_img, (28, 28))
    resized_img = resized_img.astype(np.float32) / 255.0
    resized_img = np.expand_dims(resized_img, axis=-1)  # Add channel
    resized_img = np.expand_dims(resized_img, axis=0)   # Add batch

    # Predict
    pred = model.predict(resized_img, verbose=0)
    pred_class = np.argmax(pred, axis=1)[0]
    predicted_char = class_names[pred_class]

    expression += predicted_char #+" "

    # Optional: Draw rectangles
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)

# Show prediction result
try:
    val = eval(expression)
except:
    val ="Invalid Syntax"

plt.figure(figsize=(10, 4))
plt.imshow(img, cmap='gray')
plt.title(f"Predicted Expression: {expression} Value: {val}")
plt.axis("off")
plt.show()

print("Predicted Expression:", expression )
