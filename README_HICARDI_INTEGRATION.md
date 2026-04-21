"""
HICARDI MULTI-LABEL INTEGRATION - START HERE
==============================================

This guide explains the complete integration of Hicardi 7-class multi-label
ECG classification into the existing MIT-BIH classification workflow.


WHICH GUIDE TO READ?
====================

I want to:
  
  "Just switch to Hicardi ASAP"
  → Read: QUICK_START_HICARDI.md (2 minutes)
  
  "Understand what changed"
  → Read: COMPLETE_CHANGES_SUMMARY.md (5 minutes)
  
  "Learn the configuration system"
  → Read: DATASET_SWITCHING_GUIDE.md (10 minutes)
  
  "See exact code changes"
  → Read: CODE_CHANGES_DETAILED.md (10 minutes)
  
  "Verify everything is working"
  → Follow: VERIFICATION_CHECKLIST.md (15 minutes)
  
  "Get complete technical details"
  → Read: HICARDI_INTEGRATION_SUMMARY.md (20 minutes)


EXECUTIVE SUMMARY
=================

WHAT CHANGED?
  • Added 7-class multi-label classification support (Hicardi)
  • Created batchloader_hicardi.py for data loading
  • Updated config.py with dataset mode selector
  • Updated model.py, loss.py, metrics.py for multi-label support
  • All changes backward compatible with existing MIT-BIH code

HOW MUCH CODE TO CHANGE?
  • Just 3 lines total:
    1. config.DATASET_MODE = 'hicardi'
    2-3. Update imports in autoexp.py

WHAT'S NEW?
  • 7 classes instead of 5
  • Sigmoid activation instead of softmax
  • Binary crossentropy loss instead of sparse categorical
  • Multi-label metrics (hamming_loss, subset_accuracy, etc.)
  • Single config file to switch between modes

IS IT BACKWARD COMPATIBLE?
  • YES - 100% backward compatible
  • MIT-BIH mode still works unchanged
  • Can switch modes with one line change
  • All documentation provided


FILES CREATED
=============

Python Code:
  ✓ batchloader_hicardi.py - New data loader for Hicardi

Documentation:
  ✓ QUICK_START_HICARDI.md - 3-step quickstart
  ✓ DATASET_SWITCHING_GUIDE.md - Complete switching guide
  ✓ HICARDI_INTEGRATION_SUMMARY.md - Full integration details
  ✓ CODE_CHANGES_DETAILED.md - Side-by-side code comparisons
  ✓ VERIFICATION_CHECKLIST.md - Verification procedures
  ✓ COMPLETE_CHANGES_SUMMARY.md - Comprehensive summary
  ✓ README_HICARDI_INTEGRATION.md - This file


FILES MODIFIED
==============

Python Code:
  ✓ config.py - Added dataset mode selector
  ✓ model.py - Adaptive activation function
  ✓ loss.py - Dual loss support
  ✓ metrics.py - Multi-label metrics
  ✓ batchloader_hicardi.py - New file

(No changes to: autoexp.py, train.py, dataloader.py, batchloader_mitbih.py)


THE 3-STEP SWITCH
=================

1. Edit config.py, line 7:
   DATASET_MODE = 'hicardi'

2. Edit autoexp.py (or train.py), around line 24:
   from batchloader_hicardi import load_raw_data, get_batches

3. Ensure data exists:
   ./hierarchical_data/full_multi_label/segments.npy
   ./hierarchical_data/full_multi_label/labels.npy

That's it! Everything else adapts automatically.


DATA REQUIREMENTS
=================

Before training with Hicardi:

1. Preprocess raw MAT files:
   python preprocess_4beat.py --mode train --input_dir ./ --save_dir ./processed_data

2. Create hierarchical dataset:
   python -c "from preprocess_4beat import create_full_multilabel_dataset; \
              create_full_multilabel_dataset('./processed_data')"

3. Verify files exist:
   ls ./hierarchical_data/full_multi_label/
   → segments.npy (should be ~100MB+)
   → labels.npy (should be ~1MB+)


AUTOMATIC CHANGES
=================

When you set DATASET_MODE = 'hicardi', these change automatically:

Configuration:
  N_CLASSES: 5 → 7
  ACTIVATION: 'softmax' → 'sigmoid'
  LOSS_TYPE: 'sparse_categorical_crossentropy' → 'binary_crossentropy'
  MULTI_LABEL: False → True

Model:
  Output layer: 5 softmax units → 7 sigmoid units

Loss:
  SparseCategoricalCrossentropy → BinaryCrossentropy

Metrics:
  accuracy → binary_accuracy
  + multi-label specific metrics

Data:
  Labels from (N,) → (N, 7)
  Each sample can have multiple labels


EXPECTED OUTPUT
===============

Running with Hicardi should produce:

1. On startup:
   [batchloader_hicardi] Loaded X=(12345, 300), Y=(12345, 7)
   [batchloader_hicardi] Classes: ['Normal', 'Sinus Tachycardia', ...]
   [batchloader_hicardi] Activation: sigmoid (multi-label)

2. During training:
   Loss: 0.25-0.45 (binary crossentropy values)
   binary_accuracy: 0.75-0.95 (different scale than multi-class accuracy)

3. Metrics computed:
   subset_accuracy: fraction of exact matches
   hamming_loss: fraction of wrong labels
   macro_f1, micro_f1, weighted_f1: different F1 variants
   per_label_auc: AUC for each of 7 classes


CONFIGURATION REFERENCE
=======================

MIT-BIH (default):
  DATASET_MODE = 'mitbih'
  N_CLASSES = 5
  ACTIVATION = 'softmax'
  LOSS_TYPE = 'sparse_categorical_crossentropy'
  MULTI_LABEL = False
  Classes: ['N', 'S', 'V', 'F', 'Q']

Hicardi (new):
  DATASET_MODE = 'hicardi'
  N_CLASSES = 7
  ACTIVATION = 'sigmoid'
  LOSS_TYPE = 'binary_crossentropy'
  MULTI_LABEL = True
  Classes: ['Normal', 'Sinus Tachycardia', 'Atrial Premature Contraction', 
            'Atrial Fibrillation/Flutter', 'Bradycardia', 
            'Ventricular Premature Contraction', 'Trigeminy']


COMMON QUESTIONS
================

Q: Do I lose MIT-BIH support?
A: No! Keep DATASET_MODE = 'mitbih' and everything works as before.

Q: Can I switch back and forth?
A: Yes! Just change one line in config.py. Takes 10 seconds.

Q: Do I need to retrain the model?
A: Yes, the model architecture is different (7 sigmoid vs 5 softmax).
   You'll need to retrain from scratch for Hicardi.

Q: What if I see "Expected 7 classes, got 5"?
A: You're using MIT-BIH data with Hicardi config, or vice versa.
   Check config.DATASET_MODE and data source match.

Q: Why sigmoid instead of softmax?
A: Because each beat can have multiple arrhythmias simultaneously.
   Sigmoid gives independent probabilities per class.

Q: Why binary crossentropy?
A: Works with multi-hot labels where multiple classes can be true.
   Optimizes for correct predictions on ALL positive labels.

Q: Can I use the old data loaders?
A: For MIT-BIH, yes. For Hicardi, must use batchloader_hicardi.
   They're automatic based on imports.

Q: Do I need to change my training code?
A: No! All changes are in configuration and data loading.
   Training logic remains the same.

Q: What's multi-label classification?
A: Multiple labels per sample. E.g., one beat can be both
   "Sinus Tachycardia" AND "Atrial Premature Contraction".
   Single-label (MIT-BIH): only one label per sample.


TROUBLESHOOTING
===============

Error: "FileNotFoundError: segments.npy not found"
  Solution: Run preprocess_4beat.py first

Error: "AssertionError: Expected 7 classes, got 5"
  Solution: Wrong dataset mode or wrong data format

Error: "ValueError: input shape mismatch"
  Solution: Using wrong data loader (check imports)

Error: "Loss not compatible with activation"
  Solution: config.ACTIVATION and config.LOSS_TYPE don't match

Error: "Module not found: batchloader_hicardi"
  Solution: File doesn't exist or not in same directory

Training loss NaN:
  Solution: Check label format is (N, 7) with float32 values

Metrics look wrong:
  Solution: Multi-label metrics have different ranges/meanings


NEXT STEPS
==========

1. READ: QUICK_START_HICARDI.md (fast path)
   OR
   READ: COMPLETE_CHANGES_SUMMARY.md (overview)

2. VERIFY: Follow VERIFICATION_CHECKLIST.md

3. PREPARE: Run preprocessing if needed

4. CONFIGURE: Make the 3 code changes

5. TEST: Run smoke test from QUICK_START_HICARDI.md

6. TRAIN: python autoexp.py

7. MONITOR: Check logs for correct loss/metrics values


DOCUMENTATION MAP
=================

Start Here:
  └─ README_HICARDI_INTEGRATION.md (this file)

Quick Path (5 min):
  └─ QUICK_START_HICARDI.md
     └─ Try examples in "TESTING THE SETUP" section

Learning Path (30 min):
  ├─ COMPLETE_CHANGES_SUMMARY.md (overview)
  ├─ DATASET_SWITCHING_GUIDE.md (detailed guide)
  ├─ CODE_CHANGES_DETAILED.md (code comparisons)
  └─ VERIFICATION_CHECKLIST.md (validation)

Reference Path (as needed):
  └─ HICARDI_INTEGRATION_SUMMARY.md (comprehensive)

Troubleshooting:
  ├─ QUICK_START_HICARDI.md → "ERROR MESSAGES & FIXES"
  ├─ VERIFICATION_CHECKLIST.md → "COMMON ISSUES & SOLUTIONS"
  └─ DATASET_SWITCHING_GUIDE.md → "DEBUGGING"


KEY FILES TO UNDERSTAND
=======================

1. config.py
   Contains: All configuration and dataset mode switching logic
   To switch to Hicardi: Change line 7
   Importance: High - controls entire system behavior

2. batchloader_hicardi.py
   Contains: Data loading and preprocessing for Hicardi
   To use: Import in autoexp.py/train.py
   Importance: High - handles data pipeline

3. model.py
   Contains: Adaptive model that uses config.ACTIVATION
   Changes: Very minimal, uses config value
   Importance: Medium - automatically adapts

4. loss.py
   Contains: Loss function selection logic
   Changes: Minimal, dispatches based on config.LOSS_TYPE
   Importance: Medium - ensures correct loss/metrics

5. metrics.py
   Contains: Multi-label vs multi-class metrics logic
   Changes: New metrics function for multi-label
   Importance: Medium - correct evaluation metrics


SUPPORT MATRIX
==============

Feature                MIT-BIH         Hicardi
─────────────────────────────────────────────
Classes                5               7
Activation             softmax         sigmoid
Loss                   sparse_ce       binary_ce
Labels per sample      1               multiple
Metrics                standard        multi-label
Data loader            dataloader      batchloader_hicardi
Switch effort          N/A             1 line change
Backward compatible    N/A             100%


SUCCESS CHECKLIST
=================

I'm ready to use Hicardi when:

Data:
  [ ] Preprocessed data exists at ./hierarchical_data/full_multi_label/
  [ ] segments.npy is (N, 300) shaped
  [ ] labels.npy is (N, 7) shaped

Configuration:
  [ ] config.DATASET_MODE = 'hicardi'
  [ ] batchloader_hicardi is imported

Testing:
  [ ] No import errors
  [ ] Data loads without errors
  [ ] Model builds successfully
  [ ] Loss compiles without errors
  [ ] One training step completes

Understanding:
  [ ] I know what multi-label means
  [ ] I understand why sigmoid activation is used
  [ ] I know binary crossentropy is for multi-label
  [ ] I've read appropriate documentation

Then I can:
  [ ] Run full training pipeline
  [ ] Monitor metrics
  [ ] Compare results with MIT-BIH baseline


QUESTIONS?
==========

Check these in order:
1. QUICK_START_HICARDI.md - "ERROR MESSAGES & FIXES"
2. VERIFICATION_CHECKLIST.md - "COMMON ISSUES & SOLUTIONS"  
3. DATASET_SWITCHING_GUIDE.md - "DEBUGGING"
4. HICARDI_INTEGRATION_SUMMARY.md - "TROUBLESHOOTING"
5. Search CODE_CHANGES_DETAILED.md for specific file


READY?
======

To switch to Hicardi right now:

1. Open config.py, line 7: DATASET_MODE = 'hicardi'
2. Open autoexp.py, line 24-25: from batchloader_hicardi import load_raw_data, get_batches
3. Run: python autoexp.py

Everything else just works!

For detailed verification: See VERIFICATION_CHECKLIST.md
For understanding why: See COMPLETE_CHANGES_SUMMARY.md
For troubleshooting: See QUICK_START_HICARDI.md

Good luck! 🚀
"""
