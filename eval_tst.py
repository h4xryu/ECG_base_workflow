"""
eval_tst.py — 저장된 가중치로 평가만 수행

Usage:
    python eval_tst.py
    python eval_tst.py --exp 20260423_090231_CATNet_sparse_ce_adam_none
    python eval_tst.py --weights results/.../model.weights.h5
    python eval_tst.py --model   results/.../ecg_model.h5
"""
import os
import argparse
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import config
from model   import build_model
from loss    import compile_model
from modules import ResidualUBlock, CATNet, ChannelAttention

EXP_NAME = '20260423_090231_CATNet_sparse_ce_adam_none'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp',     default=EXP_NAME)
    p.add_argument('--weights', default=None)
    p.add_argument('--model',   default=None)
    return p.parse_args()


def load_model(args) -> tf.keras.Model:
    exp_dir      = os.path.join(config.SAVE_DIR, args.exp)
    model_path   = args.model   or os.path.join(exp_dir, 'ecg_model.h5')
    weights_path = args.weights or os.path.join(exp_dir, 'model.weights.h5')

    if os.path.exists(model_path):
        print(f'[load] 전체 모델: {model_path}')
        return tf.keras.models.load_model(
            model_path,
            custom_objects={'ChannelAttention': ChannelAttention,
                            'ResidualUBlock': ResidualUBlock, 'CATNet': CATNet},
        )

    print(f'[load] 가중치: {weights_path}')
    model = build_model()
    compile_model(model)
    model(np.zeros((1, config.HICARDI_WINDOW_SIZE, 1), dtype=np.float32), training=False)
    model.load_weights(weights_path)
    return model


def main():
    from batchloader_hicardi import load_raw_data, get_tf_datasets
    from eval import full_eval_hicardi

    args  = parse_args()
    model = load_model(args)
    model.summary()

    X_mmap, Y              = load_raw_data(cache_dir=config.HICARDI_DB_ROOT)
    _, _, _, te_idx        = get_tf_datasets(X_mmap, Y)
    full_eval_hicardi(model, X_mmap, Y, te_idx, exp_name=args.exp)


if __name__ == '__main__':
    main()
