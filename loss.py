import tensorflow as tf
import config


def get_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)


def get_loss():
    return tf.keras.losses.SparseCategoricalCrossentropy()


def compile_model(model):
    model.compile(
        optimizer=get_optimizer(),
        loss=get_loss(),
        metrics=['accuracy'],
    )
    return model
