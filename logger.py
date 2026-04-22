"""
Advanced TensorBoard logger using tensorboardX SummaryWriter.

Logs:
  - Per-epoch scalars  (loss, acc, macro/weighted/per-class metrics, AUROC, AUPRC)
  - Weight histograms  (every N epochs)
  - ECG sample images  (waveform grid)
  - Confusion matrix   (rendered as image)
  - Model graph        (TF trace export to a sub-dir)
  - Model summary      (as text)

Install: pip install tensorboardX Pillow
"""

import io
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn
import tensorflow as tf
from tensorboardX import SummaryWriter
from PIL import Image

import config


class TrainingLogger:
    def __init__(self, log_dir: str, exp_name: str = ''):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir  = log_dir
        self.exp_name = exp_name
        self.writer   = SummaryWriter(log_dir=log_dir, comment=exp_name)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fig_to_chw(fig) -> np.ndarray:
        """Render a matplotlib figure to (3, H, W) uint8 numpy array."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        img = np.array(Image.open(buf).convert('RGB'))   # (H, W, 3)
        plt.close(fig)
        return img.transpose(2, 0, 1)                    # (3, H, W)

    # ── Scalar logging ────────────────────────────────────────────────────────

    def log_scalars(self, epoch: int, loss: float, acc: float, phase: str = 'train'):
        """Lightweight per-epoch logging (loss + accuracy only)."""
        p = phase.capitalize()
        self.writer.add_scalar(f'{p}/Loss',     loss, epoch)
        self.writer.add_scalar(f'{p}/Accuracy', acc,  epoch)

    def log_epoch(self, epoch: int, loss: float, metrics: dict, phase: str = 'train'):
        """Full per-epoch logging including all sklearn metrics."""
        p = phase.capitalize()
        self.writer.add_scalar(f'{p}/Loss', loss, epoch)

        if config.MULTI_LABEL:
            self._log_epoch_multilabel(epoch, p, metrics)
        else:
            self._log_epoch_multiclass(epoch, p, metrics)

    def _log_epoch_multiclass(self, epoch: int, p: str, metrics: dict):
        self.writer.add_scalar(f'{p}/Accuracy/overall', metrics['acc'], epoch)

        for key, tag in [('macro_precision',   'Macro/Precision'),
                         ('macro_recall',      'Macro/Recall'),
                         ('macro_f1',          'Macro/F1'),
                         ('macro_specificity', 'Macro/Specificity'),
                         ('macro_auroc',       'Macro/AUROC'),
                         ('macro_auprc',       'Macro/AUPRC')]:
            self.writer.add_scalar(f'{p}/{tag}', metrics[key], epoch)

        for key, tag in [('w_precision',   'Weighted/Precision'),
                         ('w_recall',      'Weighted/Recall'),
                         ('w_f1',          'Weighted/F1'),
                         ('w_specificity', 'Weighted/Specificity'),
                         ('w_auroc',       'Weighted/AUROC'),
                         ('w_auprc',       'Weighted/AUPRC')]:
            self.writer.add_scalar(f'{p}/{tag}', metrics[key], epoch)

        for i, name in enumerate(config.CLASS_NAMES):
            self.writer.add_scalar(f'{p}/PerClass/{name}/Accuracy',    metrics['pc_acc'][i], epoch)
            self.writer.add_scalar(f'{p}/PerClass/{name}/Sensitivity', metrics['pc_se'][i],  epoch)
            self.writer.add_scalar(f'{p}/PerClass/{name}/Specificity', metrics['pc_sp'][i],  epoch)
            self.writer.add_scalar(f'{p}/PerClass/{name}/Precision',   metrics['pc_pr'][i],  epoch)
            self.writer.add_scalar(f'{p}/PerClass/{name}/F1',          metrics['pc_f1'][i],  epoch)

    def _log_epoch_multilabel(self, epoch: int, p: str, metrics: dict):
        self.writer.add_scalar(f'{p}/Accuracy/subset_exact', metrics['subset_accuracy'], epoch)
        self.writer.add_scalar(f'{p}/HammingLoss',           metrics['hamming_loss'],    epoch)

        for key, tag in [('macro_precision', 'Macro/Precision'),
                         ('macro_recall',    'Macro/Recall'),
                         ('macro_f1',        'Macro/F1'),
                         ('micro_precision', 'Micro/Precision'),
                         ('micro_recall',    'Micro/Recall'),
                         ('micro_f1',        'Micro/F1'),
                         ('w_precision',     'Weighted/Precision'),
                         ('w_recall',        'Weighted/Recall'),
                         ('w_f1',            'Weighted/F1')]:
            self.writer.add_scalar(f'{p}/{tag}', metrics[key], epoch)

        for i, name in enumerate(config.CLASS_NAMES):
            self.writer.add_scalar(f'{p}/PerClass/{name}/Precision', metrics['pc_precision'][i], epoch)
            self.writer.add_scalar(f'{p}/PerClass/{name}/Recall',    metrics['pc_recall'][i],    epoch)
            self.writer.add_scalar(f'{p}/PerClass/{name}/F1',        metrics['pc_f1'][i],        epoch)
            if metrics['per_label_auc']:
                self.writer.add_scalar(f'{p}/PerClass/{name}/AUROC', metrics['per_label_auc'][i], epoch)

    # ── Weight histograms ─────────────────────────────────────────────────────

    def log_histograms(self, model, epoch: int):
        for layer in model.layers:
            for w in layer.weights:
                tag = w.name.replace(':', '_').replace('/', '_')
                self.writer.add_histogram(tag, w.numpy(), epoch)

    # ── Model graph (TF trace) ────────────────────────────────────────────────

    def log_model_graph(self, model, sample_input: np.ndarray):
        graph_dir = os.path.join(self.log_dir, 'graph')
        os.makedirs(graph_dir, exist_ok=True)
        tf_writer = tf.summary.create_file_writer(graph_dir)

        @tf.function
        def _trace(x):
            return model(x, training=False)

        tf.summary.trace_on(graph=True, profiler=False)
        _trace(tf.constant(sample_input, dtype=tf.float32))
        with tf_writer.as_default():
            tf.summary.trace_export('model_graph', step=0, profiler_outdir=graph_dir)
        tf_writer.flush()

    # ── Model summary as text ─────────────────────────────────────────────────

    def log_model_summary(self, model):
        lines = []
        model.summary(print_fn=lines.append)
        self.writer.add_text('model/summary', '\n'.join(lines), 0)

    # ── ECG waveform image grid ───────────────────────────────────────────────

    def log_ecg_samples(self, tag: str, signals: np.ndarray,
                        labels: np.ndarray, step: int, n: int = 5):
        n = min(n, len(signals))
        fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n))
        fig.patch.set_facecolor('#1a1a2e')
        if n == 1:
            axes = [axes]
        colors_list = list(config.CLASS_COLORS.values())
        for i in range(n):
            lbl = labels[i]
            if np.ndim(lbl) > 0:  # multi-label: binary vector
                active = np.where(np.asarray(lbl) > 0.5)[0]
                title = ', '.join(config.CLASS_NAMES[j] for j in active) if len(active) else 'None'
                color = colors_list[active[0] % len(colors_list)] if len(active) else colors_list[0]
            else:  # single-label: scalar index
                idx = int(lbl)
                title = config.CLASS_NAMES[idx]
                color = colors_list[idx % len(colors_list)]
            axes[i].plot(signals[i], linewidth=0.9, color=color)
            axes[i].set_title(title, color='white', fontsize=8)
            axes[i].set_facecolor('#16213e')
            axes[i].tick_params(colors='#888888', labelsize=7)
        plt.tight_layout()
        self.writer.add_image(tag, self._fig_to_chw(fig), step)

    # ── Confusion matrix image ────────────────────────────────────────────────

    def log_confusion_matrix(self, y_true, y_pred, step: int, phase: str = 'eval'):
        from sklearn.metrics import confusion_matrix
        cm  = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=config.CLASS_NAMES,
                        yticklabels=config.CLASS_NAMES, ax=ax)
        ax.set_xlabel('Predicted', color='white')
        ax.set_ylabel('True',      color='white')
        ax.tick_params(colors='white')
        plt.tight_layout()
        self.writer.add_image(f'{phase}/ConfusionMatrix', self._fig_to_chw(fig), step)

    # ── Close ─────────────────────────────────────────────────────────────────

    def close(self):
        self.writer.close()
