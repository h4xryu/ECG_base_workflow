"""
preprocess.py — Raw Hicardi .mat → segments.npy / labels.npy

Usage:
    python preprocess.py
    python preprocess.py --data_root /path/to/mat_files --out_dir ./hierarchical_data/full_multi_label

Reads:  *.mat files from data_root  (dECG, final_flag, LeadOff, data_lost, Rpk_label)
Writes: out_dir/segments.npy  (N, WINDOW_SIZE)   float32, z-score normalised
        out_dir/labels.npy    (N, N_CLASSES)      float32, multi-hot
                              classes = config.HICARDI_CLASS_NAMES
                              columns = config.HICARDI_TARGET_LABELS (final_flag indices)
"""

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from pathlib import Path

import config
from dataloader import load_holter_mat


def run(data_root: str, out_dir: str) -> None:
    print(f'[preprocess] data_root : {data_root}')
    print(f'[preprocess] out_dir   : {out_dir}')
    print(f'[preprocess] target labels (final_flag cols): {config.HICARDI_TARGET_LABELS}')
    print(f'[preprocess] class names  : {config.HICARDI_CLASS_NAMES}')
    print()

    X, Y = load_holter_mat(data_root=data_root)

    print(f'\n[preprocess] Extracted  X={X.shape}  Y={Y.shape}')
    print(f'[preprocess] Label dtype: {Y.dtype}')
    print(f'[preprocess] Positive counts per class:')
    for name, cnt in zip(config.HICARDI_CLASS_NAMES, Y.sum(axis=0).astype(int)):
        print(f'             {name:<35s} {cnt:>7,}')

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    seg_file = out_path / 'segments.npy'
    lbl_file = out_path / 'labels.npy'
    np.save(seg_file, X)
    np.save(lbl_file, Y)

    print(f'\n[preprocess] Saved → {seg_file}')
    print(f'[preprocess] Saved → {lbl_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Hicardi .mat files')
    parser.add_argument(
        '--data_root',
        default=config.DATA_ROOT,
        help=f'Directory containing .mat files (default: {config.DATA_ROOT})',
    )
    parser.add_argument(
        '--out_dir',
        default='./data/hicardi',
        help='Output directory for segments.npy and labels.npy',
    )
    args = parser.parse_args()
    run(args.data_root, args.out_dir)
