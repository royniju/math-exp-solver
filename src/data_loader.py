import os
import cv2
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

def get_characters(label_csv):
    df = pd.read_csv(label_csv)
    all_text = ''.join(df['label'].values)
    characters = sorted(list(set(all_text)))
    return characters

def text_to_labels(text, char_to_num):
    return [char_to_num[char] for char in text]

def load_ctc_data(image_dir, label_csv, img_width, img_height, characters, max_label_len):
    df = pd.read_csv(label_csv)
    images = []
    labels = []

    char_to_num = {char: idx for idx, char in enumerate(characters)}

    for _, row in df.iterrows():
        image_path = os.path.join(image_dir, row['filename'])
        label = row['label']

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.transpose(img, (1, 0, 2))  # (width, height, 1)

        images.append(img)
        labels.append(text_to_labels(label, char_to_num))

    X = np.array(images)
    y = pad_sequences(labels, maxlen=max_label_len, padding='post', value=len(characters))

    input_lengths = np.ones((len(X), 1)) * (img_width // 4)  # Based on downsampling
    label_lengths = np.array([[len(label)] for label in labels])

    return X, y, input_lengths, label_lengths
