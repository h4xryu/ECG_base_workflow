"""
batchloader_hicardi.py — Memory-mapped lazy-loading pipeline for Hicardi ECG.

Instead of loading the full .npy arrays into RAM, this module:
  1. Opens segments.npy with mmap_mode='r' — file stays on disk, only accessed
     pages are paged in by the OS on demand.
  2. Splits only the index array (not the data) into train/test.
  3. Builds tf.data pipelines that fetch individual samples via tf.py_function,
     so GPU compute and disk I/O overlap via prefetch(AUTOTUNE).

Public API
----------
    load_raw_data()           -> (X_mmap, Y)
    get_tf_datasets(X_mmap, Y) -> (train_ds, val_ds, tr_idx, te_idx)
    get_batches(X, Y)          -> (X_tr, X_te, y_tr, y_te)  [legacy, small datasets]
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

import config

_DEFAULT_CACHE_DIR = './data/hicardi'
TARGET_LENGTH      = config.HICARDI_WINDOW_SIZE   # 800
N_CLASSES          = config.HICARDI_N_CLASSES     # 9
TEST_SIZE          = config.TEST_SIZE              # 0.2
RANDOM_SEED        = config.RANDOM_SEED            # 104


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
import os
def load_raw_data(cache_dir='./mezoo_db'):
    """
    Returns (X_mmap, Y):
        X_mmap : np.memmap  (N, TARGET_LENGTH)  float32 — disk-backed, not copied to RAM
        Y      : np.ndarray (N, N_CLASSES)      float32 — fully loaded (labels are small)

    Cache hit  → mmap-opens segments.npy, fully loads labels.npy.
    Cache miss → scans all .mat files under config.HICARDI_DB_ROOT recursively,
                 saves segments.npy / labels.npy, then mmap-opens.
    """
    # cache_dir = Path(cache_dir or _DEFAULT_CACHE_DIR)
    seg_path = f"{cache_dir}/hicardi_segments.npy"
    lbl_path  = f"{cache_dir}/hicardi_labels.npy"
    print(seg_path)
    print(lbl_path)
    # if not (seg_path.exists() and lbl_path.exists()):
    #     print(f'[batchloader_hicardi] Cache miss — building from {config.HICARDI_DB_ROOT}')
    #     from dataloader import load_holter_mat
    #     cache_dir.mkdir(parents=True, exist_ok=True)
    #     X_full, Y_full = load_holter_mat(config.HICARDI_DB_ROOT)
    #     # np.save(str(seg_path), X_full)
    #     np.save(str(lbl_path), Y_full)
    #     del X_full, Y_full

    print(f'[batchloader_hicardi] Cache hit — loading from {cache_dir}')
    X_mmap = np.load(str(seg_path), mmap_mode='r')
    Y      = (np.load(str(lbl_path)) > 0).astype(np.float32)

    assert X_mmap.ndim == 2 and X_mmap.shape[1] == TARGET_LENGTH, \
        f"Unexpected segment shape {X_mmap.shape}; expected (N, {TARGET_LENGTH}). " \
        f"Delete cache and re-run."
    assert Y.shape == (X_mmap.shape[0], N_CLASSES), \
        f"Label shape mismatch: X={X_mmap.shape}, Y={Y.shape}"

    print(f'[batchloader_hicardi] X={X_mmap.shape}  Y={Y.shape}')
    print(f'[batchloader_hicardi] Classes: {config.CLASS_NAMES}')
    return X_mmap, Y


def get_tf_datasets(X_mmap, Y, batch_size=None):
    """
    Split by index → lazy tf.data pipeline that fetches from mmap on demand.

    No segment data is materialised into RAM.  Only the integer index arrays
    and the label array (Y) are held in memory — both are tiny relative to X.

    Returns
    -------
    train_ds : tf.data.Dataset  — batched, shuffled, prefetched
    val_ds   : tf.data.Dataset  — batched, prefetched
    tr_idx   : np.ndarray[int32]
    te_idx   : np.ndarray[int32]
    """
    batch_size = batch_size or config.BATCH_SIZE
    idx = np.arange(len(Y), dtype=np.int32)

    tr_idx, te_idx = train_test_split(
        idx,
        test_size    = TEST_SIZE,
        random_state = RANDOM_SEED,
        shuffle      = True,
    )

    print(f'[batchloader_hicardi] Lazy pipeline: '
          f'train={len(tr_idx)}  val={len(te_idx)} …', flush=True)

    # Y fits in RAM (N × 9 float32 ≈ 235 MB for 6.5 M samples)
    n_pos_tr = Y[tr_idx].sum(axis=0).astype(int)
    n_pos_te = Y[te_idx].sum(axis=0).astype(int)
    for name, tr, te in zip(config.CLASS_NAMES, n_pos_tr, n_pos_te):
        print(f'  {name:<35}  train={tr:6d}  val={te:5d}')

    _seg = X_mmap  # mmap reference — stays on disk

    def _fetch(i):
        i = int(i.numpy())
        x = np.array(_seg[i], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.reshape(TARGET_LENGTH, 1), Y[i]

    def _fetch_tf(i):
        x, y = tf.py_function(_fetch, [i], Tout=[tf.float32, tf.float32])
        x.set_shape((TARGET_LENGTH, 1))
        y.set_shape((N_CLASSES,))
        return x, y

    def make_ds(indices, shuffle):
        ds = tf.data.Dataset.from_tensor_slices(indices)
        if shuffle:
            # buffer holds integer indices only (4 bytes each) — negligible RAM
            ds = ds.shuffle(len(indices), reshuffle_each_iteration=True)
        ds = ds.map(_fetch_tf, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return make_ds(tr_idx, True), make_ds(te_idx, False), tr_idx, te_idx


# ─────────────────────────────────────────────────────────────────────────────
# Chunk-based inference (avoids cudnn DEVICE_TYPE_INVALID on py_function ds)
# ─────────────────────────────────────────────────────────────────────────────

def predict_from_mmap(model, X_mmap, te_idx, batch_size=None):
    """
    Run model inference directly from mmap one batch at a time.

    Why not model.predict(val_ds)?
    model.predict compiles a brand-new predict_step graph.  When the dataset
    is backed by tf.py_function the output tensors carry no concrete device
    placement, causing cudnn to report DEVICE_TYPE_INVALID and crash.

    Calling model(numpy_chunk, training=False) gives TF a fully concrete
    CPU tensor; it inserts an explicit H2D copy before the GPU conv, which
    the cudnn autotuner handles correctly.

    Peak RAM = one batch (batch_size × TARGET_LENGTH × 4 bytes ≈ 400 KB).
    """
    batch_size = batch_size or config.BATCH_SIZE
    n          = len(te_idx)
    y_proba    = np.empty((n, N_CLASSES), dtype=np.float32)

    for start in range(0, n, batch_size):
        chunk_idx = te_idx[start : start + batch_size]
        X_chunk   = np.array(X_mmap[chunk_idx], dtype=np.float32).reshape(-1, TARGET_LENGTH, 1)
        np.nan_to_num(X_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        end               = start + len(chunk_idx)
        y_proba[start:end] = model(X_chunk, training=False).numpy()

    return y_proba


# ─────────────────────────────────────────────────────────────────────────────
# Legacy: full-array API (small datasets or non-hicardi code)
# ─────────────────────────────────────────────────────────────────────────────

def get_batches(X, Y):
    """
    Legacy API: materialises full train/test arrays into RAM.
    For large Hicardi data use get_tf_datasets() instead.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        np.asarray(X), Y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_SEED,
        shuffle      = True,
        stratify     = None,
    )
    X_tr = X_tr.reshape(-1, TARGET_LENGTH, 1).astype(np.float32)
    X_te = X_te.reshape(-1, TARGET_LENGTH, 1).astype(np.float32)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)
    print(f'[batchloader_hicardi] Train: X_tr={X_tr.shape}, y_tr={y_tr.shape}')
    print(f'[batchloader_hicardi] Test:  X_te={X_te.shape}, y_te={y_te.shape}')
    n_pos_tr = np.sum(y_tr, axis=0).astype(int)
    n_pos_te = np.sum(y_te, axis=0).astype(int)
    print(f'[batchloader_hicardi] Train class dist: {n_pos_tr}')
    print(f'[batchloader_hicardi] Test  class dist: {n_pos_te}')
    return X_tr, X_te, y_tr, y_te
