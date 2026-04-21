import tensorflow as tf
from modules import CATNet, ResidualUBlock
import config


def build_model():
    """
    Build classification model with configurable output activation.
    
    - Multi-class (MIT-BIH):   softmax,  5 classes
    - Multi-label (Hicardi):   sigmoid,  7 classes
    """
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(config.WINDOW_SIZE, 1)),
        tf.keras.layers.Conv1D(30, kernel_size=15, strides=2, padding='same', name='input_conv'),

        ResidualUBlock(out_ch=180, mid_ch=30, layers=3, name='residual_u_3'),
        ResidualUBlock(out_ch=180, mid_ch=30, layers=2, name='residual_u_2'),

        # Classifier head (adaptive activation based on dataset)
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(config.N_CLASSES, activation=config.ACTIVATION),
    ])


