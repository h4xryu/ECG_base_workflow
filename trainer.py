"""
Custom training loop with tf.GradientTape.

Subclass Trainer and override train_step / val_step to add:
  - label smoothing / class-weighted loss
  - auxiliary losses
  - gradient clipping        → override train_step, call self._clip_and_apply(grads)
  - mixup / cutmix
  - mixed precision          → wrap with tf.keras.mixed_precision
  - learning-rate scheduling → attach a scheduler and call it in on_epoch_end()

Usage
-----
trainer = Trainer(model, optimizer, loss_fn)
history = trainer.fit(X_tr, y_tr, X_val, y_val,
                      epochs=30, batch_size=128,
                      logger=logger,
                      weights_path=config.WEIGHTS_PATH)
"""

import numpy as np
import tensorflow as tf
import config


# ── Keras-compatible history object ──────────────────────────────────────────

class History:
    """Mirrors the structure of tf.keras.callbacks.History for eval.py."""

    def __init__(self):
        self.history: dict[str, list] = {
            'loss': [], 'accuracy': [],
            'val_loss': [], 'val_accuracy': [],
        }

    def record(self, tr_loss: float, tr_acc: float,
               vl_loss: float, vl_acc: float):
        self.history['loss'].append(tr_loss)
        self.history['accuracy'].append(tr_acc)
        self.history['val_loss'].append(vl_loss)
        self.history['val_accuracy'].append(vl_acc)


# ── Base Trainer ──────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model     = model
        self.optimizer = optimizer
        self.loss_fn   = loss_fn

    # ── Override these ────────────────────────────────────────────────────────

    @tf.function
    def train_step(self, X_batch, y_batch):
        if config.MULTI_LABEL:
            y_batch = tf.cast(y_batch > 0, tf.float32)
        with tf.GradientTape() as tape:
            probs = self.model(X_batch, training=True)
            loss  = self.loss_fn(y_batch, probs)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, probs

    @tf.function
    def val_step(self, X_batch, y_batch):
        if config.MULTI_LABEL:
            y_batch = tf.cast(y_batch > 0, tf.float32)
        probs = self.model(X_batch, training=False)
        loss  = self.loss_fn(y_batch, probs)
        return loss, probs

    # ── Hook called at the end of each epoch (override freely) ───────────────

    def on_epoch_end(self, epoch: int, tr_loss: float, tr_acc: float,
                     vl_loss: float, vl_acc: float):
        pass

    # ── Main loop ─────────────────────────────────────────────────────────────

    def fit(self,
            X_train, y_train,
            X_val,   y_val,
            epochs:        int,
            batch_size:    int,
            logger=None,
            weights_path:  str  = None,
            hist_freq:     int  = 5,
            full_metrics_freq: int = 0) -> History:
        """
        Parameters
        ----------
        logger             : TrainingLogger — pass None to skip TensorBoard.
        weights_path       : saves best val_acc checkpoint here.
        hist_freq          : log weight histograms every N epochs.
        full_metrics_freq  : compute full sklearn metrics (AUROC, per-class…)
                             on val set every N epochs; 0 = scalars only.
        """
        history    = History()
        best_acc   = 0.0

        tr_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                 .shuffle(min(len(X_train), 10_000))
                 .batch(batch_size)
                 .prefetch(tf.data.AUTOTUNE))

        vl_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                 .batch(batch_size)
                 .prefetch(tf.data.AUTOTUNE))

        for epoch in range(epochs):

            # ── Train ─────────────────────────────────────────────────────
            tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0
            for Xb, yb in tr_ds:
                loss, probs = self.train_step(Xb, yb)
                n            = len(Xb)
                tr_loss_sum += float(loss) * n
                if config.MULTI_LABEL:
                    preds_b  = (probs.numpy() > 0.5).astype(int)
                    yb_np    = yb.numpy().astype(int)
                    tr_correct += int(np.sum(np.all(preds_b == yb_np, axis=1)))
                else:
                    tr_correct += int(tf.reduce_sum(
                        tf.cast(tf.argmax(probs, 1) == tf.cast(yb, tf.int64), tf.int32)))
                tr_total += n

            # ── Validation ────────────────────────────────────────────────
            vl_loss_sum, vl_correct, vl_total = 0.0, 0, 0
            all_preds, all_proba, all_true = [], [], []
            for Xb, yb in vl_ds:
                loss, probs = self.val_step(Xb, yb)
                proba        = probs.numpy()
                yb_np        = yb.numpy().astype(int)
                n            = len(Xb)
                vl_loss_sum += float(loss) * n
                if config.MULTI_LABEL:
                    preds_b  = (proba > 0.5).astype(int)
                    vl_correct += int(np.sum(np.all(preds_b == yb_np, axis=1)))
                    all_preds.extend(preds_b)
                else:
                    vl_correct += int(tf.reduce_sum(
                        tf.cast(tf.argmax(probs, 1) == tf.cast(yb, tf.int64), tf.int32)))
                    all_preds.extend(np.argmax(proba, axis=-1))
                vl_total    += n
                all_proba.extend(proba)
                all_true.extend(yb_np)

            tr_loss = tr_loss_sum / tr_total;  tr_acc = tr_correct / tr_total
            vl_loss = vl_loss_sum / vl_total;  vl_acc = vl_correct / vl_total
            history.record(tr_loss, tr_acc, vl_loss, vl_acc)

            print(f'Epoch {epoch+1:3d}/{epochs}  '
                  f'loss={tr_loss:.4f}  acc={tr_acc:.4f}  '
                  f'val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}')

            # ── Checkpoint ───────────────────────────────────────────────
            if weights_path and vl_acc > best_acc:
                best_acc = vl_acc
                self.model.save_weights(weights_path)

            # ── Logging ──────────────────────────────────────────────────
            if logger:
                logger.log_scalars(epoch, tr_loss, tr_acc, 'train')

                if full_metrics_freq > 0 and (epoch + 1) % full_metrics_freq == 0:
                    from metrics import compute_metrics
                    m = compute_metrics(np.array(all_true),
                                        np.array(all_preds),
                                        np.array(all_proba))
                    logger.log_epoch(epoch, vl_loss, m, 'valid')
                else:
                    logger.log_scalars(epoch, vl_loss, vl_acc, 'valid')

                if (epoch + 1) % hist_freq == 0:
                    logger.log_histograms(self.model, epoch)

            self.on_epoch_end(epoch, tr_loss, tr_acc, vl_loss, vl_acc)

        return history
