import numpy as np
import pywt
import wfdb
from os.path import join as osj
import config


# ── Denoising ────────────────────────────────────────────────────────────────

def denoise(data):
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cD1 = coeffs[-1]
    threshold = (np.median(np.abs(cD1)) / 0.6745) * np.sqrt(2 * np.log(len(cD1)))
    coeffs[-1].fill(0)
    coeffs[-2].fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    return pywt.waverec(coeffs=coeffs, wavelet='db5')


# ── Per-patient segmentation ──────────────────────────────────────────────────

def load_patient(patient_id, X_list, Y_list):
    _, info = wfdb.io.rdsamp(osj(config.DATA_ROOT, patient_id))
    channel = info['sig_name'][0]

    record = wfdb.rdrecord(osj(config.DATA_ROOT, patient_id), channel_names=[channel])
    rdata  = denoise(record.p_signal.flatten())

    ann    = wfdb.rdann(osj(config.DATA_ROOT, patient_id), 'atr')
    r_locs = ann.sample
    r_syms = ann.symbol

    for i in range(2, len(r_syms) - 3):
        try:
            label   = config.BEAT_TYPES.index(r_syms[i])
            segment = rdata[r_locs[i] - config.WINDOW_LEFT : r_locs[i] + config.WINDOW_RIGHT]
            if len(segment) == config.WINDOW_SIZE:
                X_list.append(segment)
                Y_list.append(label)
        except ValueError:
            continue


# ── Full dataset loading (15-class labels) ────────────────────────────────────

def load_raw_data():
    X, Y = [], []
    for pid in config.PATIENT_IDS:
        load_patient(pid, X, Y)

    X = np.array(X, dtype=np.float32).reshape(-1, config.WINDOW_SIZE)
    Y = np.array(Y, dtype=np.float32)

    perm = np.random.permutation(len(X))
    return X[perm], Y[perm]


# ── 15-class → 5-class (AAMI) conversion ─────────────────────────────────────

def to_5class(Y):
    Y5 = np.empty_like(Y)
    for i, y in enumerate(Y):
        if   0 <= y <= 4:  Y5[i] = 0   # N
        elif 5 <= y <= 8:  Y5[i] = 1   # S
        elif 9 <= y <= 10: Y5[i] = 2   # V
        elif y == 11:      Y5[i] = 3   # F
        else:              Y5[i] = 4   # Q
    return Y5.astype(np.int64)
