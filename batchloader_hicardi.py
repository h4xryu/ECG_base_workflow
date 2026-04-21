"""
batchloader_hicardi.py — Data loader for Hicardi multi-label ECG classification.

Drop-in replacement for dataloader + batchloader_mitbih when using Hicardi dataset.

Usage in autoexp.py / train.py:
    from batchloader_hicardi import load_raw_data, get_batches

Interface (identical to existing loaders):
    load_raw_data(data_dir=None) -> X (N, TARGET_LENGTH), Y (N, n_classes)
    get_batches(X, Y)            -> X_tr, X_te, y_tr, y_te  (shape: N, TARGET_LENGTH, 1)

Data source: full_multi_label/ directory produced by create_full_multilabel_dataset()
             (see preprocess_4beat.py)
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import config


# ────────────────────────────────────────────────────────────────────────────
# Configuration (can be overridden by passing data_dir explicitly)
# ────────────────────────────────────────────────────────────────────────────

_DEFAULT_DATA_DIR = './hierarchical_data/full_multi_label'
TARGET_LENGTH     = config.WINDOW_SIZE   # 300 samples per segment
TEST_SIZE         = config.TEST_SIZE     # 0.2
RANDOM_SEED       = config.RANDOM_SEED   # 104


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def load_raw_data(data_dir=None):
    """
    Load full_multi_label dataset produced by create_full_multilabel_dataset().
    
    Data shape:
        X: (N, TARGET_LENGTH)     — 1D ECG segments (z-score normalized)
        Y: (N, 7)                 — multi-hot labels (7 arrhythmia classes)
    
    Args:
        data_dir: Path to full_multi_label/ directory.
                  Defaults to './hierarchical_data/full_multi_label'
    
    Returns:
        X : np.ndarray  (N, TARGET_LENGTH)  — z-score normalized ECG segments
        Y : np.ndarray  (N, 7)              — multi-hot binary labels
    
    Raises:
        FileNotFoundError: If segments.npy or labels.npy not found
    """
    data_dir = Path(data_dir) if data_dir else Path(_DEFAULT_DATA_DIR)
    
    seg_file = data_dir / 'segments.npy'
    lbl_file = data_dir / 'labels.npy'
    
    if not seg_file.exists():
        raise FileNotFoundError(f"segments.npy not found at {seg_file}")
    if not lbl_file.exists():
        raise FileNotFoundError(f"labels.npy not found at {lbl_file}")
    
    X = np.load(seg_file)
    Y = np.load(lbl_file)
    
    assert X.ndim == 2, f"Expected 2D segments, got shape {X.shape}"
    assert Y.ndim == 2, f"Expected 2D labels, got shape {Y.shape}"
    assert X.shape[0] == Y.shape[0], f"Shape mismatch: X={X.shape}, Y={Y.shape}"
    assert Y.shape[1] == config.N_CLASSES, \
        f"Expected {config.N_CLASSES} classes, got {Y.shape[1]} from data"
    
    print(f'[batchloader_hicardi] Loaded X={X.shape}, Y={Y.shape} from {data_dir}')
    print(f'[batchloader_hicardi] Classes: {config.CLASS_NAMES}')
    print(f'[batchloader_hicardi] Activation: {config.ACTIVATION} (multi-label)')
    
    return X, Y


def get_batches(X, Y):
    """
    Train/test split then reshape to Conv1D input (N, TARGET_LENGTH, 1).
    
    Args:
        X : np.ndarray  (N, TARGET_LENGTH)  — ECG segments
        Y : np.ndarray  (N, 7)              — multi-hot labels
    
    Returns:
        X_tr : np.ndarray  (N_tr, TARGET_LENGTH, 1)  — float32
        X_te : np.ndarray  (N_te, TARGET_LENGTH, 1)  — float32
        y_tr : np.ndarray  (N_tr, 7)                 — float32
        y_te : np.ndarray  (N_te, 7)                 — float32
    
    Notes:
        - Uses stratify=None (random split) to avoid complexity with multi-label
        - Y is not stratified; consider MultiLabelStratifiedShuffleSplit if needed
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=None  # Multi-label stratification is complex; use random split
    )
    
    # Reshape to Conv1D input (N, TARGET_LENGTH, 1)
    X_tr = X_tr.reshape(-1, TARGET_LENGTH, 1).astype(np.float32)
    X_te = X_te.reshape(-1, TARGET_LENGTH, 1).astype(np.float32)
    
    # Ensure labels are float32 (for binary crossentropy)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)
    
    print(f'[batchloader_hicardi] Train: X_tr={X_tr.shape}, y_tr={y_tr.shape}')
    print(f'[batchloader_hicardi] Test:  X_te={X_te.shape}, y_te={y_te.shape}')
    
    # Class distribution summary
    n_positive_tr = np.sum(y_tr, axis=0)
    n_positive_te = np.sum(y_te, axis=0)
    print(f'[batchloader_hicardi] Train class distribution: {n_positive_tr.astype(int)}')
    print(f'[batchloader_hicardi] Test class distribution:  {n_positive_te.astype(int)}')
    
    return X_tr, X_te, y_tr, y_te
