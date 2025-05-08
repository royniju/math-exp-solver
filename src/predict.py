import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os

from model import CharCNN  # Assuming you saved the model definition in model.py
import pickle

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharCNN(num_classes=16)  # Replace with your class count
model.load_state_dict(torch.load("char_cnn.pth", map_location=device))
model.to(device)
model.eval()

# Load label map (index to character)
with open("label_map.pkl", "rb") as f:
    idx_to_char = pickle.load(f)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

def segment_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Find contours (connected components)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes and sort by x-coordinate (left to right)
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    boxes = sorted(boxes, key=lambda b: b[0])

    chars = []
    for x, y, w, h in boxes:
        char_img = gray[y:y+h, x:x+w]
        pil_img = Image.fromarray(char_img)
        chars.append(pil_img)

    return chars

def predict_expression(image_path):
    chars = segment_image(image_path)
    expression = ""

    for char_img in chars:
        img_tensor = transform(char_img).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            expression += idx_to_char[pred_idx]
    
    return expression

# Predict from test image
expression = predict_expression("C:/Users/nijuk/Documents/GitHub/math-exp-solver/src/test.png")
print("Predicted Expression:", expression)
