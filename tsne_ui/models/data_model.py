"""
DataModel — Observer pattern via Qt signals.

Responsibilities:
  - Scan results/ folder for valid experiments (those with tsne_data.npz).
  - Load .npz on demand and emit experiment_loaded.
  - Track the selected sample index and emit sample_selected.
"""

import os

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from constants import RESULTS_ROOT


class DataModel(QObject):
    experiment_loaded = pyqtSignal(dict)  # full data dict
    sample_selected   = pyqtSignal(dict)  # single-sample dict

    def __init__(self):
        super().__init__()
        self._data: dict | None = None

    # ── Experiment discovery ──────────────────────────────────────────────────

    def list_experiments(self) -> list[str]:
        if not os.path.isdir(RESULTS_ROOT):
            return []
        return sorted(
            name for name in os.listdir(RESULTS_ROOT)
            if os.path.isfile(os.path.join(RESULTS_ROOT, name, 'tsne_data.npz'))
        )

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_experiment(self, exp_name: str):
        if not exp_name:
            return
        path = os.path.join(RESULTS_ROOT, exp_name, 'tsne_data.npz')
        raw  = np.load(path, allow_pickle=True)
        self._data = {
            'embeddings':    raw['embeddings'],
            'labels':        raw['labels'].astype(int),
            'predictions':   raw['predictions'].astype(int),
            'probabilities': raw['probabilities'].astype(float),
            'samples':       raw['samples'].astype(float),
        }
        self.experiment_loaded.emit(self._data)

    # ── Sample selection ──────────────────────────────────────────────────────

    def select_sample(self, idx: int):
        if self._data is None:
            return
        self.sample_selected.emit({
            'signal':        self._data['samples'][idx],
            'probabilities': self._data['probabilities'][idx],
            'true_label':    int(self._data['labels'][idx]),
            'pred_label':    int(self._data['predictions'][idx]),
        })
