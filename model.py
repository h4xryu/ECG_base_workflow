import tensorflow as tf
from modules import ResidualUBlock
import config


def build_model():
    """
    ResUNet classifier: Conv1D stem → 2× ResidualUBlock → GlobalMaxPool → Dense.

    - Multi-class (MIT-BIH):  softmax,  5 classes
    - Multi-label (Hicardi):  sigmoid,  9 classes
    """
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(config.HICARDI_WINDOW_SIZE, 1)),
        tf.keras.layers.Conv1D(180, kernel_size=15, strides=1, padding='same', name='input_conv'),

        ResidualUBlock(out_ch=128, mid_ch=32, layers=3, name='residual_u_3'),
        ResidualUBlock(out_ch=128, mid_ch=32, layers=2, name='residual_u_2'),

        # Classifier head (adaptive activation based on dataset)
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(config.N_CLASSES, activation=config.ACTIVATION),
    ])


