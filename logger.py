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
import platform

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn
import tensorflow as tf
from tensorboardX import SummaryWriter
from PIL import Image

import config


def _resolve_log_dir(log_dir: str, exp_name: str) -> str:
    """
    TensorboardX's background writer thread crashes with OSError errno 5 when
    writing to a Windows NTFS filesystem mounted under WSL (/mnt/...).
    Redirect to /tmp in that case so the writer always targets a Linux fs.
    """
    if platform.system() == 'Linux' and os.path.abspath(log_dir).startswith('/mnt/'):
        safe = f'/tmp/tb_logs/{exp_name}'
        print(f'[logger] Windows mount detected — redirecting TensorBoard logs to {safe}')
        return safe
    return log_dir


class TrainingLogger:
    def __init__(self, log_dir: str, exp_name: str = ''):
        log_dir = _resolve_log_dir(log_dir, exp_name)
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir   = log_dir
        self.exp_name  = exp_name
        self._disabled = False
        try:
            self.writer = SummaryWriter(log_dir=log_dir, comment=exp_name)
        except Exception as e:
            print(f'[logger] TensorboardX unavailable ({e}); logging disabled.')
            self.writer    = None
            self._disabled = True

    def _write(self, fn, *args, **kwargs):
        """Call fn(*args) on the writer; disable logging on first I/O error."""
        if self._disabled or self.writer is None:
            return
        try:
            fn(*args, **kwargs)
        except OSError as e:
            print(f'[logger] Write error ({e}); TensorBoard logging disabled for this run.')
            self._disabled = True
        except Exception:
            pass  # non-fatal; skip silently

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
        self._write(self.writer.add_scalar, f'{p}/Loss',     loss, epoch)
        self._write(self.writer.add_scalar, f'{p}/Accuracy', acc,  epoch)

    def log_epoch(self, epoch: int, loss: float, metrics: dict, phase: str = 'train'):
        """Full per-epoch logging including all sklearn metrics."""
        p = phase.capitalize()
        self._write(self.writer.add_scalar, f'{p}/Loss', loss, epoch)

        if config.MULTI_LABEL:
            self._log_epoch_multilabel(epoch, p, metrics)
        else:
            self._log_epoch_multiclass(epoch, p, metrics)

    def _log_epoch_multiclass(self, epoch: int, p: str, metrics: dict):
        self._write(self.writer.add_scalar, f'{p}/Accuracy/overall', metrics['acc'], epoch)

        for key, tag in [('macro_precision',   'Macro/Precision'),
                         ('macro_recall',      'Macro/Recall'),
                         ('macro_f1',          'Macro/F1'),
                         ('macro_specificity', 'Macro/Specificity'),
                         ('macro_auroc',       'Macro/AUROC'),
                         ('macro_auprc',       'Macro/AUPRC')]:
            self._write(self.writer.add_scalar, f'{p}/{tag}', metrics[key], epoch)

        for key, tag in [('w_precision',   'Weighted/Precision'),
                         ('w_recall',      'Weighted/Recall'),
                         ('w_f1',          'Weighted/F1'),
                         ('w_specificity', 'Weighted/Specificity'),
                         ('w_auroc',       'Weighted/AUROC'),
                         ('w_auprc',       'Weighted/AUPRC')]:
            self._write(self.writer.add_scalar, f'{p}/{tag}', metrics[key], epoch)

        for i, name in enumerate(config.CLASS_NAMES):
            self._write(self.writer.add_scalar, f'{p}/PerClass/{name}/Accuracy',    metrics['pc_acc'][i], epoch)
            self._write(self.writer.add_scalar, f'{p}/PerClass/{name}/Sensitivity', metrics['pc_se'][i],  epoch)
            self._write(self.writer.add_scalar, f'{p}/PerClass/{name}/Specificity', metrics['pc_sp'][i],  epoch)
            self._write(self.writer.add_scalar, f'{p}/PerClass/{name}/Precision',   metrics['pc_pr'][i],  epoch)
            self._write(self.writer.add_scalar, f'{p}/PerClass/{name}/F1',          metrics['pc_f1'][i],  epoch)

    def _log_epoch_multilabel(self, epoch: int, p: str, metrics: dict):
        self._write(self.writer.add_scalar, f'{p}/Accuracy/subset_exact', metrics['subset_accuracy'], epoch)
        self._write(self.writer.add_scalar, f'{p}/HammingLoss',           metrics['hamming_loss'],    epoch)

        for key, tag in [('macro_precision', 'Macro/Precision'),
                         ('macro_recall',    'Macro/Recall'),
                         ('macro_f1',        'Macro/F1'),
                         ('micro_precision', 'Micro/Precision'),
                         ('micro_recall',    'Micro/Recall'),
                         ('micro_f1',        'Micro/F1'),
                         ('w_precision',     'Weighted/Precision'),
                         ('w_recall',        'Weighted/Recall'),
                         ('w_f1',            'Weighted/F1')]:
            self._write(self.writer.add_scalar, f'{p}/{tag}', metrics[key], epoch)

        for i, name in enumerate(config.CLASS_NAMES):
            self._write(self.writer.add_scalar, f'{p}/PerClass/{name}/Precision', metrics['pc_precision'][i], epoch)
            self._write(self.writer.add_scalar, f'{p}/PerClass/{name}/Recall',    metrics['pc_recall'][i],    epoch)
            self._write(self.writer.add_scalar, f'{p}/PerClass/{name}/F1',        metrics['pc_f1'][i],        epoch)
            if metrics['per_label_auc']:
                self._write(self.writer.add_scalar, f'{p}/PerClass/{name}/AUROC', metrics['per_label_auc'][i], epoch)

    # ── Weight histograms ─────────────────────────────────────────────────────

    def log_histograms(self, model, epoch: int):
        if self._disabled:
            return
        for layer in model.layers:
            for w in layer.weights:
                tag = w.name.replace(':', '_').replace('/', '_')
                self._write(self.writer.add_histogram, tag, w.numpy(), epoch)

    # ── Model graph (TF trace) ────────────────────────────────────────────────

    def log_model_graph(self, model, sample_input: np.ndarray):
        if self._disabled:
            return
        try:
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
        except Exception as e:
            print(f'[logger] Model graph export failed ({e}); skipping.')

    # ── Model summary as text ─────────────────────────────────────────────────

    def log_model_summary(self, model):
        lines = []
        model.summary(print_fn=lines.append)
        self._write(self.writer.add_text, 'model/summary', '\n'.join(lines), 0)

    # ── ECG waveform image grid ───────────────────────────────────────────────

    def log_ecg_samples(self, tag: str, signals: np.ndarray,
                        labels: np.ndarray, step: int, n: int = 5):
        if self._disabled:
            return
        try:
            n = min(n, len(signals))
            fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n))
            fig.patch.set_facecolor('#1a1a2e')
            if n == 1:
                axes = [axes]
            colors_list = list(config.CLASS_COLORS.values())
            for i in range(n):
                lbl = labels[i]
                if np.ndim(lbl) > 0:
                    active = np.where(np.asarray(lbl) > 0.5)[0]
                    title = ', '.join(config.CLASS_NAMES[j] for j in active) if len(active) else 'None'
                    color = colors_list[active[0] % len(colors_list)] if len(active) else colors_list[0]
                else:
                    idx = int(lbl)
                    title = config.CLASS_NAMES[idx]
                    color = colors_list[idx % len(colors_list)]
                axes[i].plot(signals[i], linewidth=0.9, color=color)
                axes[i].set_title(title, color='white', fontsize=8)
                axes[i].set_facecolor('#16213e')
                axes[i].tick_params(colors='#888888', labelsize=7)
            plt.tight_layout()
            self._write(self.writer.add_image, tag, self._fig_to_chw(fig), step)
        except Exception as e:
            print(f'[logger] ECG sample logging failed ({e}); skipping.')

    # ── Confusion matrix image ────────────────────────────────────────────────

    def log_confusion_matrix(self, y_true, y_pred, step: int, phase: str = 'eval'):
        if self._disabled:
            return
        try:
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
            self._write(self.writer.add_image, f'{phase}/ConfusionMatrix', self._fig_to_chw(fig), step)
        except Exception as e:
            print(f'[logger] Confusion matrix logging failed ({e}); skipping.')

    # ── Close ─────────────────────────────────────────────────────────────────

    def close(self):
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass
