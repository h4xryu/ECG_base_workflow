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

from constants import CLASS_COLORS, CLASS_COLORS_LIST, CLASS_NAMES, N_CLASSES

_BG_OUTER  = 'white'
_BG_INNER  = 'white'
_FG_COLOR  = '#222222'
_TICK_CLR  = '#444444'
_SPINE_CLR = '#bbbbbb'


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

        emb   = data['embeddings']
        labels= data['labels']
        probs = data['probabilities']          # (N, n_classes)
        conf  = np.max(probs, axis=1)          # max sigmoid per sample → confidence

        self._emb    = emb
        self._kdtree = KDTree(emb)

        for cls_id in range(N_CLASSES):
            mask = labels == cls_id
            color = CLASS_COLORS[cls_id]
            name  = CLASS_NAMES[cls_id]

            if mask.any():
                # alpha per point proportional to confidence (0.25–0.95)
                alphas = 0.25 + 0.70 * conf[mask]
                rgba   = np.array(
                    [(*self._hex_to_rgb(color), a) for a in alphas]
                )
                self._ax.scatter(
                    emb[mask, 0], emb[mask, 1],
                    c=rgba, s=10,
                    linewidths=0, label=name, zorder=2,
                )
            else:
                # dummy invisible point so the class appears in legend
                self._ax.scatter(
                    [], [],
                    c=color, s=10, linewidths=0,
                    label=f'{name} (0)', zorder=2, alpha=0.4,
                )

        self._style_axes()
        self._fig.tight_layout(pad=1.5)
        self.draw()

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple:
        h = hex_color.lstrip('#')
        return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    # ── Internal ──────────────────────────────────────────────────────────────

    def save_figure(self, path: str):
        self._fig.savefig(path, dpi=200, bbox_inches='tight',
                          facecolor=_BG_OUTER)

    def _draw_placeholder(self):
        self._ax.text(0.5, 0.5, 'Select an experiment',
                      color='#aaaaaa', ha='center', va='center',
                      transform=self._ax.transAxes, fontsize=14)
        self._ax.set_xticks([]); self._ax.set_yticks([])
        self.draw()

    def _style_axes(self):
        leg = self._ax.legend(
            loc='lower left', framealpha=0.92,
            labelcolor=_FG_COLOR, facecolor='white',
            edgecolor=_SPINE_CLR, markerscale=3.0,
            fontsize=9, title='Classes  (α = confidence)',
            title_fontsize=8,
        )
        leg.get_title().set_color(_FG_COLOR)
        self._ax.set_title('t-SNE', color=_FG_COLOR, fontsize=13, pad=8)
        self._ax.tick_params(colors=_TICK_CLR, labelsize=8)
        for spine in self._ax.spines.values():
            spine.set_color(_SPINE_CLR)

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
            s=160, facecolors='none', edgecolors='#333333',
            linewidths=2.0, zorder=5,
        )
        self.draw()
