# crnn_ctc_model.py
import tensorflow as tf
from tensorflow.keras import layers, Model

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_crnn_ctc_model(img_width, img_height, num_classes, max_label_len):
    input_image = layers.Input(shape=(img_width, img_height, 1), name='input_image')
    labels = layers.Input(name='label', shape=(max_label_len,), dtype='int32')
    input_length = layers.Input(name='input_length', shape=(1,), dtype='int32')
    label_length = layers.Input(name='label_length', shape=(1,), dtype='int32')

    # CNN part
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_image)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Reshape for RNN
    new_shape = (img_width // 4, (img_height // 4) * 128)
    x = layers.Reshape(target_shape=new_shape)(x)

    # RNN part
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Output layer
    y_pred = layers.Dense(num_classes + 1, activation='softmax', name='y_pred')(x)

    # CTC loss layer
    loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length]
    )

    model = Model(inputs=[input_image, labels, input_length, label_length], outputs=loss_out)

    # Separate model for inference
    prediction_model = Model(inputs=input_image, outputs=y_pred)

    return model, prediction_model
