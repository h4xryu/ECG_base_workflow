import os

CLASS_NAMES  = ['N', 'S', 'V', 'F', 'Q']
CLASS_COLORS = {
    0: '#BF878C',  # N
    1: '#8CCF97',  # S
    2: '#8AB0BF',  # V
    3: '#BFBF8C',  # F
    4: '#A88DAA',  # Q
}
CLASS_COLORS_LIST = [CLASS_COLORS[i] for i in range(5)]

# Results root = ../results relative to this file's directory
RESULTS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'results')
)
