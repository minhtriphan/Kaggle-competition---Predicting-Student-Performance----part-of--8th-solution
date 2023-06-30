import time
from custom_config import cfg
import logging
from imp import reload

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def build_log(cfg):
    reload(logging)
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s %(message)s',
        datefmt = '%H:%M:%S',
        handlers = [
            logging.FileHandler(f"train_{cfg.ver}_{time.strftime('%m%d_%H%M', time.localtime())}_seed_{cfg.seed}_folds_{''.join([str(i) for i in cfg.training_folds])}.log"),
            logging.StreamHandler()
        ]
    )
    
def print_log(message):
    if cfg.use_log:
        logging.info(message)
    else:
        print(message)
        
def metric_fn(y_pred, y_true):
    if isinstance(y_pred, pd.Series):
        return f1_score(y_true.values, y_pred.values, average = 'macro')
    elif isinstance(y_pred, np.ndarray):
        return f1_score(y_true.flatten(), y_pred.flatten(), average = 'macro')
