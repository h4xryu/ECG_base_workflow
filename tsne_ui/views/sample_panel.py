"""
SamplePanel — right-hand panel showing:
  - Status label  (True / Predicted class + correct/wrong colour)
  - ECG waveform  (top matplotlib axis)
  - Class probability bar chart  (bottom matplotlib axis)
"""

import numpy as np
from PyQt5.QtCore    import Qt
from PyQt5.QtGui     import QFont
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from constants import CLASS_COLORS_LIST, CLASS_NAMES, MULTI_LABEL

_BG_OUTER = '#1a1a2e'
_BG_INNER = '#16213e'
_TICK_CLR = '#888888'
_SPINE_CLR= '#2a2a4a'


class SamplePanel(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()

    # ── Build layout ──────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Status label
        self.status = QLabel('Click a point on the scatter plot')
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setFont(QFont('Segoe UI', 12))
        self.status.setStyleSheet('color: #888899; padding: 4px;')
        layout.addWidget(self.status)

        # Thin separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet('color: #2a2a4a;')
        layout.addWidget(sep)

        # Matplotlib figure: ECG (top 60 %) + probabilities (bottom 40 %)
        self._fig = plt.figure(figsize=(5.5, 8.5))
        self._fig.patch.set_facecolor(_BG_OUTER)
        gs = gridspec.GridSpec(2, 1, figure=self._fig,
                               height_ratios=[3, 2], hspace=0.45)
        self._ax_ecg  = self._fig.add_subplot(gs[0])
        self._ax_prob = self._fig.add_subplot(gs[1])

        self.canvas = FigureCanvasQTAgg(self._fig)
        self._style_axes()

        layout.addWidget(self.canvas, stretch=1)

    def _style_axes(self):
        for ax in (self._ax_ecg, self._ax_prob):
            ax.set_facecolor(_BG_INNER)
            ax.tick_params(colors=_TICK_CLR, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(_SPINE_CLR)

        self._ax_ecg.set_title('ECG Signal', color='white', fontsize=11, pad=6)
        self._ax_ecg.set_xlabel('Sample',    color=_TICK_CLR, fontsize=9)
        self._ax_ecg.set_ylabel('Amplitude', color=_TICK_CLR, fontsize=9)

        self._ax_prob.set_title('Class Probabilities', color='white', fontsize=11, pad=6)
        self._ax_prob.set_ylabel('Probability',        color=_TICK_CLR, fontsize=9)
        self.canvas.draw()

    # ── Public API ────────────────────────────────────────────────────────────

    def update_sample(self, sample: dict):
        signal   = sample['signal']
        probs    = sample['probabilities']
        true_lbl = sample['true_label']   # argmax index (t-SNE용 단순화)
        pred_lbl = sample['pred_label']

        if MULTI_LABEL:
            # probs: sigmoid per label → threshold 0.5
            pred_active = [CLASS_NAMES[i] for i, p in enumerate(probs) if p > 0.5]
            pred_str    = ', '.join(pred_active) if pred_active else 'None'
            true_str    = CLASS_NAMES[true_lbl]
            correct     = true_lbl in [i for i, p in enumerate(probs) if p > 0.5]
        else:
            true_str = CLASS_NAMES[true_lbl]
            pred_str = CLASS_NAMES[pred_lbl]
            correct  = true_lbl == pred_lbl

        status_color = '#7ecf8a' if correct else '#cf7e7e'
        self.status.setText(f'True: {true_str}   │   Predicted: {pred_str}')
        self.status.setStyleSheet(
            f'color: {status_color}; font-weight: bold; font-size: 13px; padding: 4px;'
        )

        self._draw_ecg(signal, true_lbl)
        self._draw_probs(probs, pred_lbl)

        self._fig.tight_layout(pad=1.2)
        self.canvas.draw()

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_ecg(self, signal: np.ndarray, true_lbl: int):
        self._ax_ecg.cla()
        self._ax_ecg.set_facecolor(_BG_INNER)
        color = CLASS_COLORS_LIST[true_lbl % len(CLASS_COLORS_LIST)]
        self._ax_ecg.plot(signal, color=color, linewidth=1.2)
        self._ax_ecg.set_title(f'ECG  (True: {CLASS_NAMES[true_lbl]})',
                                color='white', fontsize=11, pad=6)
        self._ax_ecg.set_xlabel('Sample',    color=_TICK_CLR, fontsize=9)
        self._ax_ecg.set_ylabel('Amplitude', color=_TICK_CLR, fontsize=9)
        self._ax_ecg.tick_params(colors=_TICK_CLR)
        for spine in self._ax_ecg.spines.values():
            spine.set_color(_SPINE_CLR)

    def _draw_probs(self, probs: np.ndarray, pred_lbl: int):
        self._ax_prob.cla()
        self._ax_prob.set_facecolor(_BG_INNER)

        short_names = [n.split('/')[0][:12] for n in CLASS_NAMES]  # 긴 이름 축약
        bars = self._ax_prob.bar(short_names, probs,
                                 color=CLASS_COLORS_LIST, alpha=0.82, width=0.6)

        if MULTI_LABEL:
            # threshold line
            self._ax_prob.axhline(0.5, color='#ffaa44', linewidth=1.0,
                                  linestyle='--', alpha=0.7, label='threshold')
            for i, (bar, p) in enumerate(zip(bars, probs)):
                if p > 0.5:
                    bar.set_edgecolor('white')
                    bar.set_linewidth(2.0)
        else:
            bars[pred_lbl].set_edgecolor('white')
            bars[pred_lbl].set_linewidth(2.0)

        self._ax_prob.set_ylim(0, 1.12)
        self._ax_prob.set_title('Class Probabilities', color='white', fontsize=11, pad=6)
        self._ax_prob.set_ylabel('Probability', color=_TICK_CLR, fontsize=9)
        self._ax_prob.tick_params(colors=_TICK_CLR, labelsize=7, rotation=30)
        for spine in self._ax_prob.spines.values():
            spine.set_color(_SPINE_CLR)

        for bar, p in zip(bars, probs):
            if p > 0.02:
                self._ax_prob.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f'{p:.2f}', ha='center', va='bottom',
                    color='white', fontsize=7,
                )
