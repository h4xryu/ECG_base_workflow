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
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSplitter, QVBoxLayout, QWidget,
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

        self.btn_save_tsne  = self._make_btn('💾 t-SNE')
        self.btn_save_ecg   = self._make_btn('💾 ECG')
        self.btn_save_probs = self._make_btn('💾 Probs')

        self.btn_save_tsne.clicked.connect(self._save_tsne)
        self.btn_save_ecg.clicked.connect(self._save_ecg)
        self.btn_save_probs.clicked.connect(self._save_probs)

        top_row.addWidget(title)
        top_row.addStretch()
        top_row.addWidget(self.btn_save_tsne)
        top_row.addWidget(self.btn_save_ecg)
        top_row.addWidget(self.btn_save_probs)
        top_row.addSpacing(16)
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

    # ── Save helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _make_btn(text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setFont(QFont('Segoe UI', 9))
        btn.setFixedHeight(28)
        return btn

    def _save_tsne(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save t-SNE', 'tsne.png',
            'PNG (*.png);;SVG (*.svg);;PDF (*.pdf)',
        )
        if path:
            self.tsne_canvas.save_figure(path)

    def _save_ecg(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save ECG', 'ecg.png',
            'PNG (*.png);;SVG (*.svg);;PDF (*.pdf)',
        )
        if path:
            self.sample_panel.save_ecg(path)

    def _save_probs(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Probabilities', 'probs.png',
            'PNG (*.png);;SVG (*.svg);;PDF (*.pdf)',
        )
        if path:
            self.sample_panel.save_probs(path)

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
            QPushButton            {{ background-color: #0f3460; color: #e0e0e0;
                                      border: 1px solid #444466; border-radius: 4px;
                                      padding: 3px 10px; }}
            QPushButton:hover      {{ background-color: #1a4a80; }}
            QPushButton:pressed    {{ background-color: #0a2040; }}
        """)
