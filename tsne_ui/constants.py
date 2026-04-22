import os
import sys

# parent directory (Classification_workflow/) → import config
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))
import config

CLASS_NAMES  = config.CLASS_NAMES
N_CLASSES    = config.N_CLASSES
MULTI_LABEL  = config.MULTI_LABEL

# Build color list from config (MIT-BIH or Hicardi)
_color_values    = list(config.CLASS_COLORS.values())
CLASS_COLORS     = {i: _color_values[i] for i in range(N_CLASSES)}
CLASS_COLORS_LIST= _color_values

RESULTS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'results')
)
