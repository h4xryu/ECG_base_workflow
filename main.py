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



from dataloader  import load_raw_data
from batchloader import get_batches
from train       import train
from eval        import full_eval

if __name__ == '__main__':
    X, Y                             = load_raw_data()
    X_train, X_test, y_train, y_test = get_batches(X, Y)
    model, history, exp_name         = train(X_train, y_train)
    full_eval(model, X_train, y_train, X_test, y_test, history, exp_name)
