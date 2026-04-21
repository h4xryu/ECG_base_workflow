import tensorflow as tf
from modules import CATNet, ResidualUBlock
import config


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(config.WINDOW_SIZE, 1)),
        tf.keras.layers.Conv1D(180, kernel_size=15, strides=2, padding=7, name='input_conv'),

        ResidualUBlock(out_ch=180, mid_ch=30, num_layers=3, name='residual_u_3'),
        ResidualUBlock(out_ch=180, mid_ch=30, num_layers=2, name='residual_u_2'),

        # Classifier head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(config.N_CLASSES, activation='softmax'),
    ])
