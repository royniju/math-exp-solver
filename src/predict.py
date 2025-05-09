import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from model import CharCNN  # import your trained model class
import pickle

# Load model and label map

with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)
    idx_to_char = {v: k for k, v in label_map.items()}

num_classes = len(label_map)
model = CharCNN(num_classes=num_classes)

model.load_state_dict(torch.load("char_cnn.pth", map_location="cuda" if torch.cuda.is_available() else "cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)
    idx_to_char = label_map  # because label_map is already index → char


print("label_map:", label_map)
print("idx_to_char:", idx_to_char)


# Image preprocessing
img = cv2.imread("src/test1.png", cv2.IMREAD_GRAYSCALE)
_, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours (each char)
contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bounding_boxes = [cv2.boundingRect(c) for c in contours]
bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])  # sort left to right

expression = ""

for (x, y, w, h) in bounding_boxes:
    char_img = img[y:y+h, x:x+w]
    char_img = cv2.resize(char_img, (28, 28))  # resize to model input
    char_img = char_img.astype(np.float32) / 255.0
    char_tensor = torch.tensor(char_img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(char_tensor)
        pred = torch.argmax(output, dim=1).item()
        char = idx_to_char[pred]
        expression += char

    # Draw rectangle for visualization
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)

# Display segmented characters
plt.figure(figsize=(10, 4))
plt.imshow(img, cmap='gray')
plt.title(f"Predicted Expression: {expression}")
plt.axis("off")
plt.show()

print("Predicted Expression:", expression)
