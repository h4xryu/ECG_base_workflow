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
from dataloader import load_holter_mat


# ────────────────────────────────────────────────────────────────────────────
# Configuration (can be overridden by passing data_dir explicitly)
# ────────────────────────────────────────────────────────────────────────────

_DEFAULT_DATA_DIR = './data/hicardi'
TARGET_LENGTH     = config.HICARDI_WINDOW_SIZE  # 800 samples (200 Hz × 4 s)
TEST_SIZE         = config.TEST_SIZE     # 0.2
RANDOM_SEED       = config.RANDOM_SEED   # 104


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def load_raw_data(data_dir=None):
    """
    Cache-aware loader: loads segments.npy/labels.npy if present,
    otherwise builds them from raw .mat files (config.DATA_ROOT) and saves.

    Returns:
        X : np.ndarray  (N, TARGET_LENGTH)      float32, z-score normalised
        Y : np.ndarray  (N, N_CLASSES)           float32, multi-hot
    """
    data_dir = Path(data_dir) if data_dir else Path(_DEFAULT_DATA_DIR)
    seg_file = data_dir / 'segments.npy'
    lbl_file = data_dir / 'labels.npy'

    if seg_file.exists() and lbl_file.exists():
        print(f'[batchloader_hicardi] Cache hit — loading from {data_dir}')
        X = np.load(seg_file)
        Y = (np.load(lbl_file) > 0).astype(np.float32)  # ensure binary regardless of cache age
    else:
        print(f'[batchloader_hicardi] Cache miss — building from raw .mat files in {config.DATA_ROOT}')
        X, Y = load_holter_mat(data_root=config.DATA_ROOT)
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(seg_file, X)
        np.save(lbl_file, Y)
        print(f'[batchloader_hicardi] Saved cache → {seg_file}')
        print(f'[batchloader_hicardi] Saved cache → {lbl_file}')

    assert X.ndim == 2, f"Expected 2D segments, got shape {X.shape}"
    assert Y.ndim == 2, f"Expected 2D labels, got shape {Y.shape}"
    assert X.shape[0] == Y.shape[0], f"Shape mismatch: X={X.shape}, Y={Y.shape}"
    assert Y.shape[1] == config.N_CLASSES, \
        f"Expected {config.N_CLASSES} classes, got {Y.shape[1]} — delete cache and re-run"

    print(f'[batchloader_hicardi] X={X.shape}  Y={Y.shape}')
    print(f'[batchloader_hicardi] Classes: {config.CLASS_NAMES}')
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


def prepare_and_split(data_dir=None, out_dir=None, test_size=None, random_seed=None):
    """
    Prepare train/test splits from full_multi_label dataset and save to disk.

    Args:
        data_dir:    Path to full_multi_label directory (segments.npy/labels.npy).
        out_dir:     Output directory to save splits (defaults to data_dir/splits).
        test_size:   Fraction for test split (defaults to config.TEST_SIZE).
        random_seed: Random seed for splitting (defaults to config.RANDOM_SEED).
    Returns:
        dict with paths of saved files.
    """
    from sklearn.model_selection import train_test_split

    data_dir = Path(data_dir) if data_dir else Path(_DEFAULT_DATA_DIR)
    out_dir = Path(out_dir) if out_dir else data_dir / 'splits'
    out_dir.mkdir(parents=True, exist_ok=True)

    test_size = config.TEST_SIZE if test_size is None else test_size
    random_seed = config.RANDOM_SEED if random_seed is None else random_seed

    seg_file = data_dir / 'segments.npy'
    lbl_file = data_dir / 'labels.npy'
    if not seg_file.exists() or not lbl_file.exists():
        raise FileNotFoundError(f"segments/labels not found in {data_dir}")

    X = np.load(seg_file)
    Y = np.load(lbl_file)

    X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

    # Save raw (N, L) arrays and reshaped Conv1D arrays
    np.save(out_dir / 'X_train.npy', X_tr)
    np.save(out_dir / 'X_test.npy',  X_te)
    np.save(out_dir / 'y_train.npy', y_tr)
    np.save(out_dir / 'y_test.npy',  y_te)

    # Also save Conv1D-ready versions
    np.save(out_dir / 'X_train_cnn.npy', X_tr.reshape(-1, TARGET_LENGTH, 1).astype(np.float32))
    np.save(out_dir / 'X_test_cnn.npy',  X_te.reshape(-1, TARGET_LENGTH, 1).astype(np.float32))

    print(f"Prepared splits in: {out_dir}")
    return {
        'out_dir': str(out_dir),
        'X_train': str(out_dir / 'X_train.npy'),
        'X_test':  str(out_dir / 'X_test.npy'),
        'y_train': str(out_dir / 'y_train.npy'),
        'y_test':  str(out_dir / 'y_test.npy'),
    }


if __name__ == '__main__':
    # Simple CLI to prepare splits from the default data directory
    import argparse

    parser = argparse.ArgumentParser(description='Prepare Hicardi dataset splits.')
    parser.add_argument('--data_dir', default=None, help='full_multi_label directory')
    parser.add_argument('--out_dir',  default=None, help='output splits directory')
    parser.add_argument('--test_size', type=float, default=None, help='test split fraction')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()

    prepare_and_split(data_dir=args.data_dir, out_dir=args.out_dir, test_size=args.test_size, random_seed=args.seed)
