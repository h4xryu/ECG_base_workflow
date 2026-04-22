import os
import datetime

# ============================================================
# Dataset selection: 'mitbih' or 'hicardi'
# ============================================================
DATASET_MODE = 'hicardi'  # Change to 'hicardi' for Hicardi multi-label

DATA_ROOT = './Holter01_Local_Clinic_124_records_temp'
SAVE_DIR  = './results'
RECORDS   = os.path.join(DATA_ROOT, 'RECORDS')

WINDOW_LEFT  = 99
WINDOW_RIGHT = 201
WINDOW_SIZE  = WINDOW_LEFT + WINDOW_RIGHT   # 300  (MIT-BIH single-beat)

# Hicardi 4-beat window (200 Hz × 4 s)
HICARDI_FS          = 200
HICARDI_WINDOW_SIZE = int(HICARDI_FS * 4)  # 800

PATIENT_IDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234',
]

# ============================================================
# MIT-BIH Configuration (Multi-class)
# ============================================================
MITBIH_BEAT_TYPES  = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
MITBIH_CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
MITBIH_N_CLASSES   = 5
MITBIH_ACTIVATION  = 'softmax'  # Multi-class

# ============================================================
# Hicardi Configuration (Multi-label)
# ============================================================
# Column indices into final_flag (matches Table 1 label numbers)
HICARDI_TARGET_LABELS = [0, 2, 3, 5, 6, 8, 12, 14, 16]

HICARDI_CLASS_NAMES = [
    'Normal',                        # label  0
    'Ventricular Fib./Tach.',        # label  2
    'VPC',                           # label  3 — Ventricular Premature Contraction
    'Ventricular Bigeminy',          # label  5
    'Ventricular Trigeminy',         # label  6
    'Bradycardia',                   # label  8
    'Atrial Fib./Flutter',           # label 12
    'Atrial Premature Contraction',  # label 14
    'Sinus Tachycardia',             # label 16
]
HICARDI_N_CLASSES   = 9
HICARDI_ACTIVATION  = 'sigmoid'  # Multi-label

# ============================================================
# Active configuration (based on DATASET_MODE)
# ============================================================
if DATASET_MODE == 'hicardi':
    CLASS_NAMES     = HICARDI_CLASS_NAMES
    N_CLASSES       = HICARDI_N_CLASSES
    ACTIVATION      = HICARDI_ACTIVATION
    LOSS_TYPE       = 'binary_crossentropy'  # Multi-label
    MULTI_LABEL     = True
else:  # 'mitbih'
    CLASS_NAMES     = MITBIH_CLASS_NAMES
    N_CLASSES       = MITBIH_N_CLASSES
    ACTIVATION      = MITBIH_ACTIVATION
    LOSS_TYPE       = 'sparse_categorical_crossentropy'  # Multi-class
    MULTI_LABEL     = False

# ============================================================
# Training (common for both datasets)
# ============================================================
EPOCHS           = 1 
BATCH_SIZE       = 128
LEARNING_RATE    = 1e-3
VALIDATION_SPLIT = 0.2
TEST_SIZE        = 0.2
RANDOM_SEED      = 104

# ============================================================
# Class balancing (MIT-BIH only)
# ============================================================
N_UNDERSAMPLE = 50_000   # class-0 target after undersampling
SMOTE_TARGET  = 50_000   # classes 1-4 target after SMOTE


# Experiment metadata (used in exp name & Excel)
MODEL_NAME     = 'CATNet'
OPTIMIZER_NAME = 'adam'
LOSS_NAME      = 'sparse_ce'
SCHEDULER_NAME = 'none'

# ============================================================
# Class colours (MIT-BIH)
# ============================================================
MITBIH_CLASS_COLORS = {
    'N': '#BF878C', 'S': '#8CCF97',
    'V': '#8AB0BF', 'F': '#BFBF8C', 'Q': '#A88DAA',
}

HICARDI_CLASS_COLORS = {
    'Normal':                        '#BF878C',
    'Ventricular Fib./Tach.':        '#8AB0BF',
    'VPC':                           '#8CCF97',
    'Ventricular Bigeminy':          '#BFBF8C',
    'Ventricular Trigeminy':         '#A88DAA',
    'Bradycardia':                   '#CF9F8C',
    'Atrial Fib./Flutter':           '#8CBFBF',
    'Atrial Premature Contraction':  '#BFA88C',
    'Sinus Tachycardia':             '#A8BF8C',
}

CLASS_COLORS = HICARDI_CLASS_COLORS if DATASET_MODE == 'hicardi' else MITBIH_CLASS_COLORS


def get_exp_name() -> str:
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{ts}_{MODEL_NAME}_{LOSS_NAME}_{OPTIMIZER_NAME}_{SCHEDULER_NAME}"

# Saved artefacts
MODEL_PATH   = os.path.join(os.path.join(SAVE_DIR, get_exp_name()), 'ecg_model.h5')
WEIGHTS_PATH = os.path.join(os.path.join(SAVE_DIR, get_exp_name()), 'model.weights.h5')
LOG_DIR      = os.path.join(os.path.join(SAVE_DIR, get_exp_name()), 'logs')
RESULTS_DIR  = os.path.join(SAVE_DIR, get_exp_name())
TRAIN_PKL    = os.path.join(os.path.join(SAVE_DIR, get_exp_name()), 'train_data_SMOTE.pkl')
TEST_PKL     = os.path.join(os.path.join(SAVE_DIR, get_exp_name()), 'test_data.pkl')

# t-SNE visualization
TSNE_ENABLED = True
TSNE_MAX_SAMPLES = 2000  # Limit samples for computational efficiency
TSNE_PERPLEXITY = 30     # Perplexity parameter for t-SNE