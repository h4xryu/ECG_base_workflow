"""
TSNECanvas — Matplotlib scatter embedded in Qt.

Design:
  - Emits point_clicked(int) signal when a data point is selected.
  - Uses scipy.spatial.KDTree for O(log n) nearest-neighbour lookup.
  - Highlights the selected point with a ring marker.
"""

import numpy as np
from scipy.spatial import KDTree

from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt

from constants import CLASS_COLORS, CLASS_COLORS_LIST, CLASS_NAMES

_BG_OUTER = '#1a1a2e'
_BG_INNER = '#16213e'


class TSNECanvas(FigureCanvasQTAgg):
    """Interactive t-SNE scatter plot widget."""

    point_clicked = pyqtSignal(int)

    def __init__(self):
        self._fig, self._ax = plt.subplots(figsize=(10, 9))
        super().__init__(self._fig)
        self._fig.patch.set_facecolor(_BG_OUTER)
        self._ax.set_facecolor(_BG_INNER)

        self._kdtree:   KDTree | None = None
        self._emb:      np.ndarray | None = None
        self._highlight = None

        self.mpl_connect('button_press_event', self._on_click)
        self._draw_placeholder()

    # ── Public API ────────────────────────────────────────────────────────────

    def update_data(self, data: dict):
        self._ax.cla()
        self._ax.set_facecolor(_BG_INNER)
        self._highlight = None

        emb    = data['embeddings']
        labels = data['labels']
        self._emb    = emb
        self._kdtree = KDTree(emb)

        for cls_id in range(5):
            mask = labels == cls_id
            if not mask.any():
                continue
            self._ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=CLASS_COLORS[cls_id], s=7, alpha=0.65,
                linewidths=0, label=CLASS_NAMES[cls_id], zorder=2,
            )

        self._style_axes()
        self._fig.tight_layout(pad=1.5)
        self.draw()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _draw_placeholder(self):
        self._ax.text(0.5, 0.5, 'Select an experiment',
                      color='#777799', ha='center', va='center',
                      transform=self._ax.transAxes, fontsize=14)
        self._ax.set_xticks([]); self._ax.set_yticks([])
        self.draw()

    def _style_axes(self):
        leg = self._ax.legend(
            loc='upper right', framealpha=0.25,
            labelcolor='white', facecolor=_BG_OUTER,
            edgecolor='#444466', markerscale=2.5,
        )
        self._ax.set_title('t-SNE', color='white', fontsize=13, pad=8)
        self._ax.tick_params(colors='#888888', labelsize=8)
        for spine in self._ax.spines.values():
            spine.set_color('#2a2a4a')

    def _on_click(self, event):
        if event.inaxes != self._ax or self._kdtree is None:
            return
        dist, idx = self._kdtree.query([event.xdata, event.ydata])

        # Threshold = 1.5 % of data range
        if self._emb is None:
            return
        x_span = np.ptp(self._emb[:, 0])
        y_span = np.ptp(self._emb[:, 1])
        threshold = max(x_span, y_span) * 0.015

        if dist <= threshold:
            self._show_highlight(self._emb[idx])
            self.point_clicked.emit(int(idx))

    def _show_highlight(self, xy: np.ndarray):
        if self._highlight is not None:
            self._highlight.remove()
        self._highlight = self._ax.scatter(
            [xy[0]], [xy[1]],
            s=130, facecolors='none', edgecolors='white',
            linewidths=2.0, zorder=5,
        )
        self.draw()
