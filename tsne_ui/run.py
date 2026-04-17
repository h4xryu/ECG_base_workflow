"""
Entry point for the t-SNE ECG Explorer UI.

Usage:
    cd tsne_ui
    python run.py

Requirements:
    pip install PyQt5 matplotlib scipy
"""

import os
import sys

# Must set Qt5Agg backend BEFORE any matplotlib import
import matplotlib
matplotlib.use('Qt5Agg')

# Ensure tsne_ui/ is on sys.path so absolute imports work
sys.path.insert(0, os.path.dirname(__file__))

from PyQt5.QtWidgets import QApplication

from models.data_model          import DataModel
from views.main_window          import MainWindow
from controllers.app_controller import AppController


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('t-SNE ECG Explorer')

    model      = DataModel()
    window     = MainWindow()
    _ctrl      = AppController(model, window)   # keeps controller alive

    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
