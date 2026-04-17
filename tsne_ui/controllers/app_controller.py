"""
AppController — wires DataModel ↔ Views.

Design patterns used:
  Observer  : Qt signals/slots decouple model from view completely.
  MVC       : DataModel (M) ← this class (C) → MainWindow / canvases (V).
  Strategy  : point_clicked handler can be swapped by reassigning _on_point_clicked.
"""

from models.data_model  import DataModel
from views.main_window  import MainWindow


class AppController:
    def __init__(self, model: DataModel, window: MainWindow):
        self._model  = model
        self._window = window
        self._connect()
        self._populate_experiments()

    # ── Wiring ────────────────────────────────────────────────────────────────

    def _connect(self):
        w, m = self._window, self._model

        # View → Model
        w.exp_combo.currentTextChanged.connect(m.load_experiment)
        w.tsne_canvas.point_clicked.connect(m.select_sample)

        # Model → View
        m.experiment_loaded.connect(w.tsne_canvas.update_data)
        m.sample_selected.connect(w.sample_panel.update_sample)

    # ── Initialise experiment list ────────────────────────────────────────────

    def _populate_experiments(self):
        exps = self._model.list_experiments()
        self._window.exp_combo.addItem('')        # blank default
        self._window.exp_combo.addItems(exps)
