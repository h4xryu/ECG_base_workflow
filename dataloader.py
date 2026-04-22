import numpy as np
import pywt
import wfdb
from os.path import join as osj
from pathlib import Path
from scipy.interpolate import interp1d
import mat73
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
            label   = config.MITBIH_BEAT_TYPES.index(r_syms[i])
            segment = rdata[r_locs[i] - config.WINDOW_LEFT : r_locs[i] + config.WINDOW_RIGHT]
            if len(segment) == config.WINDOW_SIZE:
                X_list.append(segment)
                Y_list.append(label)
        except ValueError:
            continue


# ── Holter MAT loader (Hicardi multi-label) ───────────────────────────────────

def _extract_4beat_segments_from_mat(mat_data):
    """Extract 4-beat ECG segments and multi-hot labels from a Hicardi .mat file."""
    dECG       = np.array(mat_data['dECG'])
    final_flag = np.array(mat_data['final_flag'])
    LeadOff    = np.array(mat_data['LeadOff']).astype(bool)
    data_lost  = np.array(mat_data['data_lost']).astype(bool)
    Rpk_label  = np.array(mat_data['Rpk_label']).astype(bool)

    ecg_mv  = (dECG - 8192) / 1000
    r_peaks = np.where(Rpk_label)[0]

    if len(r_peaks) < 5:
        return [], []

    segments, labels = [], []
    for i in range(0, len(r_peaks) - 4, 4):
        start_idx = r_peaks[i]
        end_idx   = r_peaks[i + 4]

        if np.any(LeadOff[start_idx:end_idx]) or np.any(data_lost[start_idx:end_idx]):
            continue

        if np.any(np.isin(np.argmax(final_flag[start_idx:end_idx], axis=1), [97, 98, 99, 100])):
            continue

        segment = ecg_mv[start_idx:end_idx]
        if np.any(np.isnan(segment)):
            continue

        x_orig   = np.linspace(0, 1, len(segment))
        x_target = np.linspace(0, 1, config.HICARDI_WINDOW_SIZE)
        segment  = interp1d(x_orig, segment, kind='linear')(x_target)

        mean, std = np.mean(segment), np.std(segment)
        segment   = (segment - mean) / (std + 1e-8)

        full_label = (np.max(final_flag[start_idx:end_idx], axis=0) > 0).astype(np.float32)
        seg_label  = full_label[config.HICARDI_TARGET_LABELS]
        if seg_label.sum() == 0:   # no flags → Normal
            seg_label[0] = 1.0
        segments.append(segment)
        labels.append(seg_label)

    return segments, labels


def load_holter_mat(data_root=None):
    """Load all .mat files from a Holter directory and extract 4-beat segments."""
    data_root = Path(data_root or config.DATA_ROOT)
    mat_files = sorted(data_root.glob('*.mat'))

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {data_root}")

    all_segments, all_labels = [], []
    for mat_file in mat_files:
        try:
            mat_data = mat73.loadmat(str(mat_file))
            segs, lbls = _extract_4beat_segments_from_mat(mat_data)
            all_segments.extend(segs)
            all_labels.extend(lbls)
        except Exception as e:
            print(f"  Warning: skipping {mat_file.name} — {e}")

    if not all_segments:
        raise RuntimeError("No valid segments extracted from any .mat file.")

    X = np.array(all_segments, dtype=np.float32)
    Y = np.array(all_labels,   dtype=np.float32)

    perm = np.random.permutation(len(X))
    return X[perm], Y[perm]


# ── Full dataset loading (15-class labels) ────────────────────────────────────

def load_raw_data():
    if config.DATASET_MODE == 'hicardi':
        return load_holter_mat()

    X, Y = [], []
    for pid in config.PATIENT_IDS:
        load_patient(pid, X, Y)

    X = np.array(X, dtype=np.float32).reshape(-1, config.WINDOW_SIZE)
    Y = np.array(Y, dtype=np.float32)

    perm = np.random.permutation(len(X))
    return X[perm], Y[perm]


# ── 15-class → 5-class (AAMI) conversion ─────────────────────────────────────

if __name__ == '__main__':
    import config

    print(f'Dataset mode: {config.DATASET_MODE}')
    X, Y = load_raw_data()
    print(f'\nX shape : {X.shape}   dtype={X.dtype}')
    print(f'Y shape : {Y.shape}   dtype={Y.dtype}')

    print(f'\n--- 샘플 5개 ---')
    for i in range(min(5, len(X))):
        if Y.ndim == 2:  # multi-label (Hicardi)
            active = [config.CLASS_NAMES[j] for j, v in enumerate(Y[i]) if v > 0]
            label_str = ', '.join(active) if active else 'None'
            print(f'[{i}] signal min={X[i].min():.3f} max={X[i].max():.3f} | '
                  f'label vector={Y[i].astype(int).tolist()} | active={label_str}')
        else:  # single-label (MIT-BIH)
            print(f'[{i}] signal min={X[i].min():.3f} max={X[i].max():.3f} | '
                  f'label={int(Y[i])} ({config.CLASS_NAMES[int(Y[i])]})')

    if Y.ndim == 2:
        print(f'\n--- 클래스별 양성 샘플 수 ---')
        counts = Y.sum(axis=0).astype(int)
        for name, cnt in zip(config.CLASS_NAMES, counts):
            bar = '█' * (cnt * 40 // (counts.max() or 1))
            print(f'  {name:<35} {cnt:6d}  {bar}')

        unique_vals = np.unique(Y)
        print(f'\n라벨 unique values: {unique_vals}  (should be only 0 and 1)')
    else:
        print(f'\n--- 클래스별 샘플 수 ---')
        for i, name in enumerate(config.CLASS_NAMES):
            cnt = int((Y == i).sum())
            print(f'  {name}: {cnt}')


def to_5class(Y):
    """Convert MIT-BIH 15-class integer labels to 5-class AAMI labels. Y must be 1D."""
    assert Y.ndim == 1, (
        f"to_5class expects 1D MIT-BIH label array, got shape {Y.shape}. "
        "For Hicardi multi-label data use batchloader_hicardi instead."
    )
    Y5 = np.empty(len(Y), dtype=np.int64)
    for i, y in enumerate(Y):
        y = int(y)
        if   0 <= y <= 4:  Y5[i] = 0   # N
        elif 5 <= y <= 8:  Y5[i] = 1   # S
        elif 9 <= y <= 10: Y5[i] = 2   # V
        elif y == 11:      Y5[i] = 3   # F
        else:              Y5[i] = 4   # Q
    return Y5
