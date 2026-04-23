import os
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import config

# Hicardi multi-label classification 
def run_hicardi():
    from batchloader_hicardi import load_raw_data, get_tf_datasets
    from train import train_from_datasets
    from eval  import full_eval_hicardi

    X_mmap, Y                        = load_raw_data(cache_dir=config.HICARDI_DB_ROOT)
    train_ds, val_ds, tr_idx, te_idx = get_tf_datasets(X_mmap, Y)
    model, history, exp_name         = train_from_datasets(train_ds, val_ds, X_mmap, Y, tr_idx, te_idx)
    full_eval_hicardi(model, X_mmap, Y, te_idx, history, exp_name)

# MIT-BIH multi-class classification 
def run_mitbih():
    from dataloader        import load_raw_data
    from batchloader_mitbih import get_batches
    from train import train
    from eval  import full_eval

    X, Y                             = load_raw_data()
    X_train, X_test, y_train, y_test = get_batches(X, Y)
    model, history, exp_name         = train(X_train, y_train)
    full_eval(model, X_train, y_train, X_test, y_test, history, exp_name)


if __name__ == '__main__':
    run_hicardi() if config.DATASET_MODE == 'hicardi' else run_mitbih()
