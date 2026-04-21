import tensorflow as tf
import config


def get_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)


def get_loss():
    """
    Get loss function based on dataset mode.
    
    - MIT-BIH (multi-class):   SparseCategoricalCrossentropy
    - Hicardi (multi-label):   BinaryCrossentropy
    """
    if config.LOSS_TYPE == 'binary_crossentropy':
        return tf.keras.losses.BinaryCrossentropy()
    else:  # 'sparse_categorical_crossentropy'
        return tf.keras.losses.SparseCategoricalCrossentropy()


def compile_model(model):
    """
    Compile model with appropriate loss and metrics.
    
    For multi-label, we use binary accuracy; for multi-class, regular accuracy.
    """
    if config.MULTI_LABEL:
        metrics = ['binary_accuracy']
    else:
        metrics = ['accuracy']
    
    model.compile(
        optimizer=get_optimizer(),
        loss=get_loss(),
        metrics=metrics,
    )
    return model
