import tensorflow as tf
from modules import CATNet
import config


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(config.WINDOW_SIZE, 1)),

        CATNet(),

        # Classifier head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(config.N_CLASSES, activation='softmax'),
    ])
