from custom_config import cfg, set_random_seed, LEVEL2QUESTION
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
import polars as pl
import pandas as pd

def fold_assign(df_label, df):
    split = StratifiedGroupKFold(n_splits = cfg.nfolds)
    
    df_label['fold'] = -1
    for i, (trn_idx, val_idx) in enumerate(split.split(df_label, y = df_label.correct, groups = df_label.session)):
        df_label.loc[val_idx, 'fold'] = i

    session2fold = df_label[['session', 'fold']].drop_duplicates()
    session2fold = session2fold.set_index('session').to_dict()['fold']

    if cfg.use_polar:
        df = df.with_columns(pl.col('session_id').map_dict(session2fold).alias('fold'))
    else:
        df['fold'] = df['session_id'].map(session2fold)
        
    return df_label, df, session2fold

def split_data(df, df_labels, fold = 0, level = '0-4'):
    '''
    Split the data into training/validation sets based on folds and questions
    '''
    past_level = []
    if level == '0-4':
        past_levels = ['0-4']
    elif level == '5-12':
        past_levels = ['5-12']
    else:
        past_levels = ['13-22']
        
    if cfg.use_polar:
        trn = df.filter(pl.col('fold') != fold).filter(pl.col('level_group').is_in(past_levels))
        val = df.filter(pl.col('fold') == fold).filter(pl.col('level_group').is_in(past_levels))
    else:
        trn = df[(df['fold'] != fold) & (df['level_group'].isin(past_levels))]
        val = df[(df['fold'] == fold) & (df['level_group'].isin(past_levels))]
        
    trn_labels = df_labels[(df_labels['fold'] != fold) & (df_labels['qid'].isin(LEVEL2QUESTION[level]))]
    val_labels = df_labels[(df_labels['fold'] == fold) & (df_labels['qid'].isin(LEVEL2QUESTION[level]))]
    
    return trn, val, trn_labels, val_labels
