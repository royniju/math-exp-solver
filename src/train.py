import tensorflow as tf
import numpy as np
import os

from torch import dropout

# --- Config ---
data_dir = "dataset"
img_size = (28, 28)
batch_size = 32
validation_split = 0.2
seed = 123

# --- Load Training and Validation Datasets ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=validation_split,
    subset="training",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    label_mode='int'  # Get labels as integers first
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=validation_split,
    subset="validation",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    label_mode='int'
)

# --- Number of Classes ---
class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# --- One-hot Encoding and Normalization ---
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, num_classes)
    return image, label

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

# --- Data Augmentation ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.03),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.05)
])

def augment(image, label):
    image = data_augmentation(image)
    return image, label

train_ds = train_ds.map(augment)

# --- Prefetching ---
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# --- CNN Model ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (4, 4), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (4, 4), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- Train ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50
)

# --- Save the Model ---
model.save("symbol_classifier_tf_model.keras")
