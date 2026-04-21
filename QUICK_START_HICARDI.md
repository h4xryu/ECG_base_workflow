"""
QUICK REFERENCE - HOW TO SWITCH TO HICARDI
===========================================

⚡ FASTEST WAY TO START (3 changes)
====================================

CHANGE 1: config.py (1 line)
----------------------------
Line 7:  DATASET_MODE = 'mitbih'
         ↓ CHANGE TO ↓
Line 7:  DATASET_MODE = 'hicardi'


CHANGE 2: autoexp.py (2 lines)
-------------------------------
Lines 24-25:
    from dataloader  import load_raw_data
    from batchloader_mitbih import get_batches
    
    ↓ CHANGE TO ↓
    
    from batchloader_hicardi import load_raw_data, get_batches


CHANGE 3: train.py (if using) - same as autoexp.py
---------------------------------------------------
    from dataloader  import load_raw_data
    from batchloader_mitbih import get_batches
    
    ↓ CHANGE TO ↓
    
    from batchloader_hicardi import load_raw_data, get_batches


THAT'S IT! ✓
============

After these 3 changes, the system will automatically:
  ✓ Use 7 classes instead of 5
  ✓ Use sigmoid activation instead of softmax
  ✓ Use binary_crossentropy loss
  ✓ Compute multi-label metrics
  ✓ Load from ./hierarchical_data/full_multi_label/


BEFORE RUNNING
==============

Make sure these files exist:
  □ ./hierarchical_data/full_multi_label/segments.npy  (N, 300)
  □ ./hierarchical_data/full_multi_label/labels.npy    (N, 7)

If not, run preprocessing:
  python preprocess_4beat.py --mode train
  python -c "from preprocess_4beat import create_full_multilabel_dataset; \
             create_full_multilabel_dataset('./processed_data')"


WHAT CHANGED AUTOMATICALLY
============================

In config.py:
  N_CLASSES    5 → 7
  ACTIVATION   'softmax' → 'sigmoid'
  LOSS_TYPE    'sparse_categorical_crossentropy' → 'binary_crossentropy'
  MULTI_LABEL  False → True
  CLASS_NAMES  ['N','S','V','F','Q'] → [7 Hicardi classes]

In model.py:
  Dense output layer activation: softmax → sigmoid

In loss.py:
  Loss function: SparseCategoricalCrossentropy → BinaryCrossentropy
  Metrics: accuracy → binary_accuracy

In metrics.py:
  Metrics computed: multi-label specific metrics


SWITCHING BACK TO MIT-BIH
=========================

Just 1 line change in config.py:
  DATASET_MODE = 'hicardi' → DATASET_MODE = 'mitbih'

Everything else automatically reverts!


FILE LOCATIONS
==============

MIT-BIH data:
  ./mit-bih-arrhythmia-database-1.0.0/

Hicardi data:
  ./hierarchical_data/full_multi_label/
    ├── segments.npy  (N, 300)
    ├── labels.npy    (N, 7)
    └── mapping.json


EXPECTED OUTPUT ON STARTUP
===========================

[batchloader_hicardi] Loaded X=(12345, 300), Y=(12345, 7) from ./hierarchical_data/full_multi_label
[batchloader_hicardi] Classes: ['Normal', 'Sinus Tachycardia', ...]
[batchloader_hicardi] Activation: sigmoid (multi-label)
[batchloader_hicardi] Train: X_tr=(9876, 300, 1), y_tr=(9876, 7)
[batchloader_hicardi] Test:  X_te=(2469, 300, 1), y_te=(2469, 7)
[batchloader_hicardi] Train class distribution: [1234 2345 3456 4567 1234 2345 3456]
[batchloader_hicardi] Test class distribution:  [308 586 864 1141 308 586 864]


ERROR MESSAGES & FIXES
======================

❌ FileNotFoundError: segments.npy not found
   → Run: python preprocess_4beat.py --mode train
   → Or verify path is ./hierarchical_data/full_multi_label/

❌ AssertionError: Expected 7 classes, got 5 from data
   → Data is MIT-BIH format, need Hicardi preprocessed data
   → Run full preprocessing pipeline

❌ Shape mismatch: (N, 5) vs (N, 7)
   → config.py not switched to 'hicardi'
   → Or wrong loader (using old batchloader_mitbih)

❌ Loss not compatible with activation
   → Check config.ACTIVATION and config.LOSS_TYPE match
   → For Hicardi: sigmoid + binary_crossentropy ✓


AUTOMATED SYSTEM CHANGES
========================

When DATASET_MODE changes to 'hicardi', these happen automatically:

✓ config.CLASS_NAMES updated to 7 Hicardi class names
✓ config.N_CLASSES updated to 7
✓ config.ACTIVATION set to 'sigmoid'
✓ config.LOSS_TYPE set to 'binary_crossentropy'
✓ config.MULTI_LABEL set to True
✓ model.py Dense layer uses sigmoid
✓ loss.py returns BinaryCrossentropy
✓ metrics.py computes multi-label metrics
✓ batchloader_hicardi loads correct data format


TESTING THE SETUP
=================

Quick test to verify configuration:

    import config
    print(f"Mode: {config.DATASET_MODE}")
    print(f"Classes: {config.N_CLASSES} - {config.CLASS_NAMES}")
    print(f"Activation: {config.ACTIVATION}")
    print(f"Loss: {config.LOSS_TYPE}")
    print(f"Multi-label: {config.MULTI_LABEL}")
    
    from batchloader_hicardi import load_raw_data, get_batches
    X, Y = load_raw_data()
    X_tr, X_te, y_tr, y_te = get_batches(X, Y)
    print(f"Data loaded: X_tr={X_tr.shape}, y_tr={y_tr.shape}")


NOTES
=====

- All existing MIT-BIH code continues to work unchanged
- Simply change one line (DATASET_MODE) to switch
- Multi-label means multiple arrhythmias can be true for one beat
- Sigmoid gives independent probabilities per class
- Binary CE loss optimizes for ALL labels, not just one winner
- New metrics better reflect multi-label performance
"""
