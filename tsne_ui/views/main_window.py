"""
MainWindow — top-level QMainWindow.

Layout (1550 × 870):
  ┌─ top bar ──────────────────────────────────────────────────────────────┐
  │  Title                                    Experiment: [──────────────▼]│
  ├─ separator ────────────────────────────────────────────────────────────┤
  │  TSNECanvas (left, 1000 px)  │  SamplePanel (right, 530 px)           │
  └────────────────────────────────────────────────────────────────────────┘
"""

from PyQt5.QtCore    import Qt
from PyQt5.QtGui     import QFont
from PyQt5.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QSplitter, QVBoxLayout, QWidget,
)

from views.tsne_canvas  import TSNECanvas
from views.sample_panel import SamplePanel

_DARK = '#1a1a2e'


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('t-SNE ECG Explorer')
        self.resize(1550, 870)
        self._build_ui()
        self._apply_theme()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # ── Top bar ───────────────────────────────────────────────────────
        top = QWidget()
        top_row = QHBoxLayout(top)
        top_row.setContentsMargins(0, 0, 0, 0)

        title = QLabel('t-SNE ECG Explorer')
        title.setFont(QFont('Segoe UI', 16, QFont.Bold))

        exp_label = QLabel('Experiment:')
        exp_label.setFont(QFont('Segoe UI', 11))

        self.exp_combo = QComboBox()
        self.exp_combo.setMinimumWidth(400)
        self.exp_combo.setFont(QFont('Segoe UI', 10))
        self.exp_combo.setMaxVisibleItems(20)

        top_row.addWidget(title)
        top_row.addStretch()
        top_row.addWidget(exp_label)
        top_row.addWidget(self.exp_combo)
        layout.addWidget(top)

        # ── Separator ─────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f'color: #2a2a4a;')
        layout.addWidget(sep)

        # ── Main split ────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        self.tsne_canvas  = TSNECanvas()
        self.sample_panel = SamplePanel()

        splitter.addWidget(self.tsne_canvas)
        splitter.addWidget(self.sample_panel)
        splitter.setSizes([1000, 530])
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, stretch=1)

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget   {{ background-color: {_DARK}; color: #e0e0e0; }}
            QComboBox              {{ background-color: #16213e; color: #e0e0e0;
                                      border: 1px solid #444466; border-radius: 4px;
                                      padding: 4px 8px; }}
            QComboBox QAbstractItemView {{ background-color: #16213e; color: #e0e0e0;
                                           selection-background-color: #0f3460; }}
            QComboBox::drop-down   {{ border: none; width: 22px; }}
            QSplitter::handle      {{ background-color: #2a2a4a; }}
            QLabel                 {{ color: #e0e0e0; }}
        """)
