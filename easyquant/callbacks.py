"""QAT 훈련용 Keras 콜백 — CosineRestartSchedule, LRSchedulerCallback, SnapshotSaver."""

import math
import os

import tensorflow as tf

keras = tf.keras


class CosineRestartSchedule:
    """Cosine annealing with warm restart 학습률 스케줄."""

    def __init__(self, lr_max: float, lr_min: float, cycle_length: int):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.cycle_length = cycle_length

    def __call__(self, epoch: int) -> float:
        t = epoch % self.cycle_length
        cos_v = 0.5 * (1.0 + math.cos(math.pi * t / self.cycle_length))
        return self.lr_min + (self.lr_max - self.lr_min) * cos_v


class LRSchedulerCallback(keras.callbacks.Callback):
    """에포크 시작 시 schedule(epoch)를 호출해 optimizer learning rate를 갱신한다."""

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        del logs
        lr = self.schedule(epoch)
        keras.backend.set_value(self.model.optimizer.learning_rate, lr)


class SnapshotSaver(keras.callbacks.Callback):
    """사이클 종료 에포크마다 가중치 스냅샷을 저장한다."""

    def __init__(self, save_dir: str, cycle_length: int):
        super().__init__()
        self.save_dir = save_dir
        self.cycle_length = cycle_length
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        del logs
        if (epoch + 1) % self.cycle_length == 0:
            idx = (epoch + 1) // self.cycle_length
            self.model.save_weights(
                f"{self.save_dir}/snapshot_{idx:02d}.weights.h5"
            )
