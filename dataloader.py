import numpy as np
import pywt
import wfdb
from os.path import join as osj
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import interp1d
import mat73
from tqdm import tqdm
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

    pre  = config.HICARDI_PRE
    post = config.HICARDI_POST
    sig_len = len(ecg_mv)

    segments, labels = [], []
    for i in range(0, len(r_peaks) - 4, 4):
        r_start = r_peaks[i]
        r_end   = r_peaks[i + 4]

        start_idx = r_start - pre
        end_idx   = r_end   + post

        # skip boundary segments
        if start_idx < 0 or end_idx > sig_len:
            continue

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


def _load_single_mat(mat_path_str):
    """Load one .mat file and extract segments. Top-level so it's picklable."""
    try:
        mat_data = mat73.loadmat(mat_path_str)
        segs, lbls = _extract_4beat_segments_from_mat(mat_data)
        return segs, lbls, None
    except Exception as e:
        return [], [], str(e)


def load_holter_mat(data_root=None, n_workers=2, flush_threshold=20000, flush_every_files=30):
    """Load all .mat files under data_root in parallel and extract 4-beat segments.

    Memory-safe behavior:
      - Default `n_workers` reduced to 2 to limit concurrent memory usage.
      - Accumulated segments are flushed to chunk files when they exceed
        `flush_threshold` to avoid holding too much in RAM.
      - At the end all chunks are merged into a memory-mapped output file
        which is returned as numpy arrays (safe for large datasets).
    """
    data_root = Path(data_root or config.DATA_ROOT)
    mat_files = sorted(data_root.rglob('*.mat'))

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {data_root}")

    n_workers = min(max(1, int(n_workers)), len(mat_files))
    print(f"[load_holter_mat] {len(mat_files)} files — {n_workers} threads")

    chunks_dir = data_root / '.chunks_tmp'
    chunks_dir.mkdir(exist_ok=True)
    chunk_idx = 0
    all_segments, all_labels = [], []
    n_skipped = 0
    total_count = 0

    files_since_flush = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_single_mat, str(f)): f for f in mat_files}
        with tqdm(total=len(mat_files), desc='MAT loading', unit='file', dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                segs, lbls, err = future.result()
                if err is not None:
                    n_skipped += 1
                    pbar.write(f"  skip {futures[future].name}: {err}")
                else:
                    if segs:
                        all_segments.extend(segs)
                        all_labels.extend(lbls)
                        total_count += len(segs)
                        files_since_flush += 1

                # Flush if accumulated too large OR we've processed a fixed number of files
                if len(all_segments) >= flush_threshold or (flush_every_files and files_since_flush >= flush_every_files):
                    seg_arr = np.array(all_segments, dtype=np.float32)
                    lbl_arr = np.array(all_labels, dtype=np.float32)
                    np.save(chunks_dir / f'chunk_segs_{chunk_idx}.npy', seg_arr)
                    np.save(chunks_dir / f'chunk_lbls_{chunk_idx}.npy', lbl_arr)
                    chunk_idx += 1
                    all_segments.clear()
                    all_labels.clear()
                    files_since_flush = 0

                pbar.update(1)
                pbar.set_postfix(segs=total_count, skip=n_skipped)

    # If nothing extracted
    if total_count == 0 and not (all_segments):
        raise RuntimeError("No valid segments extracted from any .mat file.")

    # Write final chunk if any remaining
    if all_segments:
        seg_arr = np.array(all_segments, dtype=np.float32)
        lbl_arr = np.array(all_labels, dtype=np.float32)
        np.save(chunks_dir / f'chunk_segs_{chunk_idx}.npy', seg_arr)
        np.save(chunks_dir / f'chunk_lbls_{chunk_idx}.npy', lbl_arr)
        chunk_idx += 1
        all_segments.clear(); all_labels.clear()

    # Merge chunks into a memory-mapped final array to avoid large RAM peaks
    chunk_files = sorted(chunks_dir.glob('chunk_segs_*.npy'))
    total_count = 0
    chunk_sizes = []
    for cf in chunk_files:
        arr = np.load(cf, mmap_mode='r')
        chunk_sizes.append(arr.shape[0])
        total_count += arr.shape[0]

    lbl_files = sorted(chunks_dir.glob('chunk_lbls_*.npy'))
    if len(lbl_files) != len(chunk_files):
        raise RuntimeError('Chunk labels/files mismatch')

    # allocate memmap outputs in data_root
    out_segs = data_root / 'hicardi_segments.npy'
    out_lbls = data_root / 'hicardi_labels.npy'
    # infer shapes
    sample_len = config.HICARDI_WINDOW_SIZE
    n_labels = len(config.HICARDI_TARGET_LABELS)

    seg_mm = np.lib.format.open_memmap(str(out_segs), mode='w+', dtype=np.float32,
                                       shape=(total_count, sample_len))
    lbl_mm = np.lib.format.open_memmap(str(out_lbls), mode='w+', dtype=np.float32,
                                       shape=(total_count, n_labels))

    ptr = 0
    for s_cf, l_cf in zip(chunk_files, lbl_files):
        s = np.load(s_cf)
        l = np.load(l_cf)
        n = s.shape[0]
        seg_mm[ptr:ptr+n] = s.astype(np.float32)
        lbl_mm[ptr:ptr+n] = l.astype(np.float32)
        ptr += n

    # cleanup chunk files
    for f in chunk_files + lbl_files:
        try:
            f.unlink()
        except Exception:
            pass
    try:
        chunks_dir.rmdir()
    except Exception:
        pass

    # shuffle indices by writing a permuted view into final files to avoid large RAM copies.
    perm = np.random.permutation(total_count)

    # create final permuted memmap files (overwrite existing out_segs/out_lbls)
    out_segs_perm = data_root / 'hicardi_segments.npy'
    out_lbls_perm = data_root / 'hicardi_labels.npy'

    perm_segs = np.lib.format.open_memmap(str(out_segs_perm), mode='w+', dtype=np.float32,
                                          shape=(total_count, sample_len))
    perm_lbls = np.lib.format.open_memmap(str(out_lbls_perm), mode='w+', dtype=np.float32,
                                          shape=(total_count, n_labels))

    # write permuted blocks in streaming manner
    block = 0
    block_size = 10_000
    for start in range(0, total_count, block_size):
        end = min(start + block_size, total_count)
        idx = perm[start:end]
        perm_segs[start:end] = seg_mm[idx]
        perm_lbls[start:end] = lbl_mm[idx]
        block += 1

    # flush and close
    try:
        perm_segs._mmap.flush()
        perm_lbls._mmap.flush()
    except Exception:
        pass

    # close original memmaps
    try:
        seg_mm._mmap.close()
        lbl_mm._mmap.close()
    except Exception:
        pass

    # Return paths to the final on-disk memmap files to avoid copying large arrays in memory
    return str(out_segs_perm), str(out_lbls_perm)


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
