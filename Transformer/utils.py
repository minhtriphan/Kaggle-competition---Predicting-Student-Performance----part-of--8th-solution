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
        
def drop_reset_sections(df: pd.DataFrame, local: bool = True) -> pd.DataFrame:
    '''
    There are some sections that the users reset their game play.
    Particularly, they start the game as usual, at level 0-4, then 5-12, then 13-22, but then they continue with 0-4 again.
    This function will drop the repeated parts. 
    Credit to https://www.kaggle.com/code/abaojiang/lb-0-694-tconv-with-4-features-training-part
    
    Parameters:
        df: pd.DataFrame
    
    Return:
        df: pd.DataFrame with events occurring at the first game play only
    '''
    if local:
        df['lv_diff'] = df.groupby('session_id').apply(lambda x: x['encoded_level'].diff().fillna(0)).values
    else:
        df['lv_diff'] = df['encoded_level'].diff().fillna(0)
    df.loc[df['lv_diff'] >= 0, 'lv_diff'] = 0
    if local:
        df['multi_game_flag'] = df.groupby('session_id')['lv_diff'].cumsum()
    else:
        df['multi_game_flag'] = df['lv_diff'].cumsum()
    multi_game_rows = df[df['multi_game_flag'] < 0].index
    print_log(f"Dropping {len(multi_game_rows)} observations, which is corresponding to {df.loc[multi_game_rows, 'session_id'].nunique()}/{df.session_id.nunique()} sections")
    df = df.drop(multi_game_rows).reset_index(drop = True)
    df.drop(['lv_diff', 'multi_game_flag'], axis = 1, inplace = True)
    return df
        
def metric_fn(y_pred, y_true):
    if isinstance(y_pred, pd.Series):
        return f1_score(y_true.values, y_pred.values, average = 'macro')
    elif isinstance(y_pred, np.ndarray):
        return f1_score(y_true.flatten(), y_pred.flatten(), average = 'macro')
