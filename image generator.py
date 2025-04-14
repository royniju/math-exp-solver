from PIL import Image, ImageDraw, ImageFont
import os
import random
import csv

# Create folders
os.makedirs("dataset/images", exist_ok=True)

# Expression components
digits = list("0123456789")
operators = ["+", "-", "×", "÷"]
font_path = "./fonts/HomemadeApple-Regular.ttf"  # You need to download and provide the path

# Expression generator
def random_expr():
    return f"{random.choice(digits)} {random.choice(operators)} {random.choice(digits)}"

# Image generator
def create_image(expression, filename):
    img = Image.new("L", (128, 64), color=255)  # grayscale image
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 42)
    draw.text((10, 10), expression, fill=0, font=font)
    img.save(filename)

# CSV file
with open("dataset/labels.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "label"])

    for i in range(500):  # Number of expressions
        expr = random_expr()
        filename = f"img_{i:03d}.png"
        filepath = os.path.join("dataset/images", filename)
        create_image(expr, filepath)
        writer.writerow([filename, expr])
