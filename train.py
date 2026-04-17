import os

import tensorflow as tf

import config
from logger  import TrainingLogger
from loss    import get_loss, get_optimizer
from model   import build_model
from modules import CATNet, ChannelAttention
from trainer import Trainer


def train(X_train, y_train):
    exp_name = config.get_exp_name()
    log_dir  = os.path.join(config.LOG_DIR, exp_name)
    logger   = TrainingLogger(log_dir=log_dir, exp_name=exp_name)

    # ── Load existing model ───────────────────────────────────────────────────
    if os.path.exists(config.MODEL_PATH):
        print('Loading existing model.')
        model = tf.keras.models.load_model(
            config.MODEL_PATH,
            custom_objects={'ChannelAttention': ChannelAttention, 'CATNet': CATNet},
        )
        logger.close()
        return model, None, exp_name

    # ── Build ─────────────────────────────────────────────────────────────────
    model = build_model()
    model.summary()
    logger.log_model_graph(model, X_train[:1])
    logger.log_model_summary(model)
    logger.log_ecg_samples('ECG/train_samples',
                            X_train[:5].squeeze(-1), y_train[:5], step=0)

    # ── Validation split ──────────────────────────────────────────────────────
    n_val            = int(len(X_train) * config.VALIDATION_SPLIT)
    X_val,  y_val    = X_train[:n_val],  y_train[:n_val]
    X_tr,   y_tr     = X_train[n_val:],  y_train[n_val:]

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(model, get_optimizer(), get_loss())
    history = trainer.fit(
        X_tr, y_tr, X_val, y_val,
        epochs        = config.EPOCHS,
        batch_size    = config.BATCH_SIZE,
        logger        = logger,
        weights_path  = config.WEIGHTS_PATH,
        hist_freq     = 1,
        full_metrics_freq = 1,   
    )

    model.save(config.MODEL_PATH)
    logger.close()
    return model, history, exp_name
