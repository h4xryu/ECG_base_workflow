"""
EXACT CODE CHANGES - SIDE BY SIDE
==================================


FILE 1: config.py
=================

BEFORE:
-------
import os
import datetime

DATA_ROOT = './mit-bih-arrhythmia-database-1.0.0'
...
BEAT_TYPES  = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
N_CLASSES   = 5


AFTER:
------
import os
import datetime

# ============================================================
# Dataset selection: 'mitbih' or 'hicardi'
# ============================================================
DATASET_MODE = 'mitbih'  # Change to 'hicardi' for Hicardi multi-label

DATA_ROOT = './mit-bih-arrhythmia-database-1.0.0'
...

# ============================================================
# MIT-BIH Configuration (Multi-class)
# ============================================================
MITBIH_BEAT_TYPES  = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
MITBIH_CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
MITBIH_N_CLASSES   = 5
MITBIH_ACTIVATION  = 'softmax'

# ============================================================
# Hicardi Configuration (Multi-label)
# ============================================================
HICARDI_CLASS_NAMES = ['Normal', 'Sinus Tachycardia', 'Atrial Premature Contraction', ...]
HICARDI_N_CLASSES   = 7
HICARDI_ACTIVATION  = 'sigmoid'

# ============================================================
# Active configuration (based on DATASET_MODE)
# ============================================================
if DATASET_MODE == 'hicardi':
    CLASS_NAMES = HICARDI_CLASS_NAMES
    N_CLASSES = HICARDI_N_CLASSES
    ACTIVATION = HICARDI_ACTIVATION
    LOSS_TYPE = 'binary_crossentropy'
    MULTI_LABEL = True
else:  # 'mitbih'
    CLASS_NAMES = MITBIH_CLASS_NAMES
    N_CLASSES = MITBIH_N_CLASSES
    ACTIVATION = MITBIH_ACTIVATION
    LOSS_TYPE = 'sparse_categorical_crossentropy'
    MULTI_LABEL = False


FILE 2: model.py
================

BEFORE:
-------
def build_model():
    return tf.keras.Sequential([
        ...
        # Classifier head
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(config.N_CLASSES, activation='softmax'),
    ])


AFTER:
------
def build_model():
    """
    Build classification model with configurable output activation.
    
    - Multi-class (MIT-BIH):   softmax,  5 classes
    - Multi-label (Hicardi):   sigmoid,  7 classes
    """
    return tf.keras.Sequential([
        ...
        # Classifier head (adaptive activation based on dataset)
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(config.N_CLASSES, activation=config.ACTIVATION),
    ])


FILE 3: loss.py
===============

BEFORE:
-------
def get_loss():
    return tf.keras.losses.SparseCategoricalCrossentropy()


def compile_model(model):
    model.compile(
        optimizer=get_optimizer(),
        loss=get_loss(),
        metrics=['accuracy'],
    )
    return model


AFTER:
------
def get_loss():
    """
    Get loss function based on dataset mode.
    
    - MIT-BIH (multi-class):   SparseCategoricalCrossentropy
    - Hicardi (multi-label):   BinaryCrossentropy
    """
    if config.LOSS_TYPE == 'binary_crossentropy':
        return tf.keras.losses.BinaryCrossentropy()
    else:  # 'sparse_categorical_crossentropy'
        return tf.keras.losses.SparseCategoricalCrossentropy()


def compile_model(model):
    """
    Compile model with appropriate loss and metrics.
    
    For multi-label, we use binary accuracy; for multi-class, regular accuracy.
    """
    if config.MULTI_LABEL:
        metrics = ['binary_accuracy']
    else:
        metrics = ['accuracy']
    
    model.compile(
        optimizer=get_optimizer(),
        loss=get_loss(),
        metrics=metrics,
    )
    return model


FILE 4: metrics.py
==================

BEFORE:
-------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    n      = config.N_CLASSES
    
    acc = accuracy_score(y_true, y_pred)
    ... (multi-class specific metrics)
    
    return { ... }


AFTER:
------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    """
    Compute classification metrics.
    
    Dispatches to multi-label or multi-class metrics based on config.MULTI_LABEL
    """
    if config.MULTI_LABEL:
        return _compute_metrics_multilabel(y_true, y_pred, y_proba)
    else:
        return _compute_metrics_multiclass(y_true, y_pred, y_proba)


def _compute_metrics_multiclass(...): 
    # Original multi-class metrics (unchanged)
    ...

def _compute_metrics_multilabel(y_true, y_pred, y_proba=None):
    # NEW: Multi-label specific metrics
    subset_acc = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
    hamming = hamming_loss(y_true, y_pred)
    
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Per-label AUC
    per_label_auc = []
    if y_proba is not None:
        for i in range(config.N_CLASSES):
            try:
                auc = roc_auc_score(y_true[:, i], y_proba[:, i])
                per_label_auc.append(auc)
            except:
                per_label_auc.append(0.0)
    
    return {
        'subset_accuracy': subset_acc,
        'hamming_loss': hamming,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'micro_f1': micro_f1,
        'per_label_auc': per_label_auc,
        ... (other metrics)
    }


FILE 5: batchloader_hicardi.py (NEW FILE)
==========================================

NEW FILE - Contents:
--------------------
"""
batchloader_hicardi.py — Data loader for Hicardi multi-label ECG classification.

Drop-in replacement for dataloader + batchloader_mitbih when using Hicardi dataset.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import config

def load_raw_data(data_dir=None):
    """Load full_multi_label dataset produced by create_full_multilabel_dataset()."""
    data_dir = Path(data_dir) if data_dir else Path('./hierarchical_data/full_multi_label')
    
    seg_file = data_dir / 'segments.npy'
    lbl_file = data_dir / 'labels.npy'
    
    if not seg_file.exists():
        raise FileNotFoundError(f"segments.npy not found at {seg_file}")
    if not lbl_file.exists():
        raise FileNotFoundError(f"labels.npy not found at {lbl_file}")
    
    X = np.load(seg_file)
    Y = np.load(lbl_file)
    
    assert Y.shape[1] == config.N_CLASSES, f"Expected {config.N_CLASSES} classes, got {Y.shape[1]}"
    
    print(f'[batchloader_hicardi] Loaded X={X.shape}, Y={Y.shape}')
    print(f'[batchloader_hicardi] Classes: {config.CLASS_NAMES}')
    print(f'[batchloader_hicardi] Activation: {config.ACTIVATION} (multi-label)')
    
    return X, Y


def get_batches(X, Y):
    """Train/test split then reshape to Conv1D input (N, TARGET_LENGTH, 1)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, Y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        shuffle=True,
        stratify=None
    )
    
    X_tr = X_tr.reshape(-1, config.WINDOW_SIZE, 1).astype(np.float32)
    X_te = X_te.reshape(-1, config.WINDOW_SIZE, 1).astype(np.float32)
    
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)
    
    print(f'[batchloader_hicardi] Train: X_tr={X_tr.shape}, y_tr={y_tr.shape}')
    print(f'[batchloader_hicardi] Test:  X_te={X_te.shape}, y_te={y_te.shape}')
    
    return X_tr, X_te, y_tr, y_te


FILE 6: autoexp.py (or train.py) - IMPORTS ONLY
================================================

BEFORE:
-------
from dataloader  import load_raw_data
from batchloader_mitbih import get_batches


AFTER:
------
from batchloader_hicardi import load_raw_data, get_batches
# (single import combines both)


SUMMARY OF CHANGES
==================

1. config.py
   - Added DATASET_MODE selector
   - Separated MIT-BIH and Hicardi configurations
   - Dynamic assignment based on mode selection
   
   Key variables now available:
   - config.DATASET_MODE: 'mitbih' or 'hicardi'
   - config.N_CLASSES: 5 or 7 (auto-selected)
   - config.ACTIVATION: 'softmax' or 'sigmoid' (auto-selected)
   - config.LOSS_TYPE: 'sparse_categorical_crossentropy' or 'binary_crossentropy' (auto-selected)
   - config.MULTI_LABEL: False or True (auto-selected)

2. model.py
   - Changed Dense layer from fixed 'softmax' to config.ACTIVATION
   - Now adapts to dataset mode automatically
   
3. loss.py
   - get_loss() checks config.LOSS_TYPE and returns appropriate loss
   - compile_model() uses correct metrics for each mode
   
4. metrics.py
   - Dispatches to multi-label or multi-class metrics
   - Supports both single-label and multi-label evaluation
   
5. batchloader_hicardi.py (NEW)
   - Loads 7-class Hicardi multi-label data
   - Integrates with config for flexibility
   - Drop-in replacement for dataloader + batchloader_mitbih
   
6. autoexp.py / train.py
   - Update imports from batchloader_hicardi
   - No other code changes needed


TO ENABLE HICARDI
=================

1. config.py:    DATASET_MODE = 'hicardi'
2. autoexp.py:   from batchloader_hicardi import load_raw_data, get_batches
3. train.py:     from batchloader_hicardi import load_raw_data, get_batches

That's it! System automatically adapts to:
- 7 classes (instead of 5)
- sigmoid activation (instead of softmax)
- binary_crossentropy loss
- multi-label metrics
- Hicardi data loading
"""
