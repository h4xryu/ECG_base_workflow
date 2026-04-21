"""
SWITCHING BETWEEN MIT-BIH (MITBIH) AND HICARDI DATASETS
========================================================

This guide shows how to switch the classification workflow between the MIT-BIH dataset
and the Hicardi dataset using the enhanced configuration system.

QUICK START - Switch to Hicardi
================================

1. In config.py, change:
   
   DATASET_MODE = 'mitbih'  # Change this line to:
   DATASET_MODE = 'hicardi'

2. In train.py or autoexp.py, change the imports:

   # OLD (MIT-BIH)
   from dataloader  import load_raw_data
   from batchloader_mitbih import get_batches

   # NEW (Hicardi)
   from batchloader_hicardi import load_raw_data, get_batches

3. Ensure the preprocessing step has been run:
   
   python preprocess_4beat.py --mode train --input_dir ./ --save_dir ./processed_data
   python preprocess_4beat.py --mode test --input_dir ./Holter_folder --save_dir ./processed_data_test


CONFIGURATION DIFFERENCES
==========================

MIT-BIH (config.py):
  - N_CLASSES = 5
  - ACTIVATION = 'softmax'
  - LOSS_TYPE = 'sparse_categorical_crossentropy'
  - MULTI_LABEL = False
  - Data loader: dataloader.py + batchloader_mitbih.py

Hicardi (config.py):
  - N_CLASSES = 7
  - ACTIVATION = 'sigmoid'
  - LOSS_TYPE = 'binary_crossentropy'
  - MULTI_LABEL = True
  - Data loader: batchloader_hicardi.py


MODEL CHANGES
=============

When DATASET_MODE = 'hicardi':
  - model.py: Last Dense layer uses sigmoid activation (multi-label)
  - loss.py: Uses BinaryCrossentropy loss
  - metrics.py: Uses multi-label specific metrics (hamming_loss, subset_accuracy)


DATA PATHS
==========

MIT-BIH:
  - Input: ./mit-bih-arrhythmia-database-1.0.0/
  - Data loader looks for preprocessed files

Hicardi:
  - Input: ./hierarchical_data/full_multi_label/
  - Expected files:
    * segments.npy  (N, 300) - normalized ECG segments
    * labels.npy    (N, 7)   - multi-hot binary labels


WORKFLOW EXAMPLE (Hicardi)
===========================

1. Preprocess raw MAT files:
   
   python preprocess_4beat.py --mode train --input_dir ./ --save_dir ./processed_data
   
   This creates:
   - ./processed_data/*_segments.npy, *_labels.npy
   - ./processed_data/class_map.json
   
2. Create hierarchical dataset:
   
   In preprocess_4beat.py, call:
   create_full_multilabel_dataset('./processed_data', './hierarchical_data')
   
   This creates:
   - ./hierarchical_data/full_multi_label/segments.npy
   - ./hierarchical_data/full_multi_label/labels.npy
   - ./hierarchical_data/full_multi_label/mapping.json

3. Set config and train:
   
   In config.py:
   DATASET_MODE = 'hicardi'
   
   In train.py/autoexp.py:
   from batchloader_hicardi import load_raw_data, get_batches
   
   python autoexp.py


METRICS DIFFERENCES
===================

MIT-BIH (Multi-class):
  - accuracy
  - macro_f1, w_f1
  - macro_auroc, w_auroc
  - per-class sensitivity, specificity, precision

Hicardi (Multi-label):
  - subset_accuracy (exact match)
  - hamming_loss
  - macro_f1, w_f1, micro_f1
  - per-label precision, recall, F1
  - per_label_auc


CLASS NAMES
===========

MIT-BIH (5 classes):
  ['N', 'S', 'V', 'F', 'Q']
  (Normal, Supraventricular ectopy, Ventricular ectopy, Fusion, Unclassifiable)

Hicardi (7 classes):
  ['Normal',
   'Sinus Tachycardia',
   'Atrial Premature Contraction',
   'Atrial Fibrillation/Flutter',
   'Bradycardia',
   'Ventricular Premature Contraction',
   'Trigeminy']


DEBUGGING
=========

If you see shape mismatch errors:
  - Check N_CLASSES in config.py matches data (5 for MIT-BIH, 7 for Hicardi)
  - Verify preprocessed data exists in the correct directory
  - Check batchloader imports match dataset mode

If loss won't compile:
  - Ensure config.ACTIVATION matches config.LOSS_TYPE
  - For Hicardi: activation='sigmoid' + loss='binary_crossentropy'
  - For MIT-BIH: activation='softmax' + loss='sparse_categorical_crossentropy'

If metrics fail:
  - Ensure config.MULTI_LABEL is correctly set
  - Check y_true and y_pred shapes match
"""
