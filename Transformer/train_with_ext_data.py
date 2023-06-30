# Python packages
import os, pickle
import numpy as np
import pandas as pd
import polars as pl

import warnings
warnings.filterwarnings('ignore')
    
# Custom packages
from custom_config import cfg, set_random_seed, NUM_COLS, TXT_COLS, LEVEL_MAP, LEVEL2QUESTION
from kfold_split import fold_assign, split_data
from feature_engineering import process_data, get_txt_cols_encoding_maps
from utils import print_log, metric_fn, build_log, drop_reset_sections
from training_utils import fit, evaluate, infer_embedding, organizing_oof

def main():    
    ################ Initial settings ################
    # Set seed
    set_random_seed(cfg.seed)
    
    # Build log
    if cfg.use_log:
        build_log(cfg)
        
    ################ Import and process the data ################
    if cfg.use_polar:
        train = pl.read_csv(os.path.join(cfg.comp_data_dir, 'train.csv'))
        train_sup = pl.read_csv(os.path.join(cfg.ext_data_dir, 'train_sup.csv'))
    else:
        train = pd.read_csv(os.path.join(cfg.comp_data_dir, 'train.csv'))
        train_sup = pd.read_csv(os.path.join(cfg.ext_data_dir, 'train_sup.csv'))
        
    train_labels = pd.read_csv(os.path.join(cfg.comp_data_dir, 'train_labels.csv'))
    train_labels_sup = pd.read_csv(os.path.join(cfg.ext_data_dir, 'train_labels_sup.csv'))
    
    train = process_data(train, only_drop = False)
    train_sup = process_data(train_sup, only_drop = False)
        
    # Process train labels
    train_labels['session'] = train_labels['session_id'].apply(lambda x: x.split('_')[0]).astype(np.int64)
    train_labels['qid'] = train_labels['session_id'].apply(lambda x: x.split('_')[1])
    
    # Process suplementary train labels
    train_labels_sup['session'] = train_labels_sup['session_id'].apply(lambda x: x.split('_')[0]).astype(np.int64)
    train_labels_sup['qid'] = train_labels_sup['session_id'].apply(lambda x: x.split('_')[1])
            
    if not cfg.done_kfold_split:
        ################ KFold split ################
        train_labels, train, session2fold = fold_assign(train_labels, train)
            
        # Store the KFold split
        os.makedirs(cfg.ext_data_dir, exist_ok = True)
        if cfg.use_polar:
            train.to_pandas().to_csv(os.path.join(cfg.ext_data_dir, 'train_fold.csv'), index = False)
        else:
            train.to_csv(os.path.join(cfg.ext_data_dir, 'train_fold.csv'), index = False)
        train_labels.to_csv(os.path.join(cfg.ext_data_dir, 'train_labels_fold.csv'), index = False)
    else:
        # Assign to folds
        session2fold = pd.read_csv(os.path.join(cfg.ext_data_dir, 'session2fold.csv'))
            
        train = train.merge(session2fold, on = 'session_id', how = 'inner')
        train_labels = train_labels.merge(session2fold, left_on = 'session', right_on = 'session_id', 
                                          how = 'left', suffixes = ('', '_')).drop(['session_id_'], axis = 1)
        
    if cfg.use_polar:
        train = train.to_pandas()
        train_sup = train_sup.to_pandas()
        
    train_sup['fold'] = -1
    train_labels_sup['fold'] = -1
    
    train_sup = train_sup[train.columns]
    train_labels_sup = train_labels_sup[train_labels.columns]
    
    train = pd.concat([train, train_sup])
    train_labels = pd.concat([train_labels, train_labels_sup])
    
    if cfg.mode in ['eval', 'infer_embedding']:
        with open(os.path.join(cfg.model_dir, cfg.ver[:-1], cfg.ver[-1], f'TXT_COL_MAPS.pkl'), 'rb') as f:
            TXT_COL_MAPS = pickle.load(f)
    
    else:
        TXT_COL_MAPS = get_txt_cols_encoding_maps(train, TXT_COLS)
        
        with open(os.path.join(cfg.model_dir, cfg.ver[:-1], cfg.ver[-1], f'TXT_COL_MAPS.pkl'), 'wb') as f:
            pickle.dump(TXT_COL_MAPS, f)
    
    for col in TXT_COL_MAPS:
        train[col] = train[col].map(TXT_COL_MAPS[col])
    
    train['encoded_level'] = train['level_group'].map(LEVEL_MAP)
    train['change_in_level'] = train.groupby('session_id')['encoded_level'].diff().fillna(0)
    
    train = drop_reset_sections(train, local = True)
    
    train['time_diff'] = train.groupby('session_id', sort = False).apply(lambda x: x['elapsed_time'].diff().fillna(0)).values
    train['time_diff'] = train['time_diff'].clip(lower = 0)
    
    if cfg.use_polar:
        train = pl.from_pandas(train)
        new_columns = add_columns_pl(train)
        train = train.with_columns(new_columns)
    
    OOFS = []
    for fold in cfg.training_folds:
        print_log(f' FOLD {fold} '.center(50, '*'))
        
        oofs = []
        
        for i, level in enumerate(['0-4', '5-12', '13-22']):
            print_log(f' Level {level} '.center(50, '-'))
            if cfg.mode == 'train':
                valid_score = fit(cfg, train, train_labels, TXT_COL_MAPS, fold = fold, level = level)
                oof = pd.read_csv(os.path.join(cfg.model_dir, cfg.ver[:-1], cfg.ver[-1], f"oof_level_{level.replace('-', '_')}_fold_{fold}.csv"), index_col = 0)
                print_log(f'Level {level}: {valid_score}')
                oofs.append(oof)
            elif cfg.mode == 'eval':
                valid_score, oof = evaluate(cfg, train, train_labels, TXT_COL_MAPS, fold = fold, level = level)
                print_log(f'Level {level}: {valid_score}')
                oofs.append(oof)
            elif cfg.mode == 'infer_embedding':
                trn_indexes, trn_embeddings, trn_predictions, val_indexes, val_embeddings, val_predictions = infer_embedding(cfg, train, train_labels, TXT_COL_MAPS, fold = fold, level = level)
                
                trn_embeddings_df = pd.DataFrame(data = trn_embeddings, index = trn_indexes)
                val_embeddings_df = pd.DataFrame(data = val_embeddings, index = val_indexes)
                
                trn_embeddings_df.to_csv(os.path.join(cfg.model_dir, cfg.ver[:-1], cfg.ver[-1], f"{cfg.ver}_trn_embeddings_level_{level.replace('-', '_')}_fold_{fold}.csv"))
                val_embeddings_df.to_csv(os.path.join(cfg.model_dir, cfg.ver[:-1], cfg.ver[-1], f"{cfg.ver}_val_embeddings_level_{level.replace('-', '_')}_fold_{fold}.csv"))
        
        if cfg.mode != 'infer_embedding':
            # Compute the fold score
            oofs = pd.concat(oofs).sort_values('session_id')
            score = metric_fn((oofs['pred_correct'] > 0.63).astype(int), oofs['correct'])
            print_log(f'Fold {fold}: {score}')
            oofs.to_csv(os.path.join(cfg.model_dir, cfg.ver[:-1], cfg.ver[-1], f'{cfg.ver}_oofs_fold_{fold}.csv'), index = False)
            
            OOFS.append(oofs)
            
    if cfg.mode != 'infer_embedding':
        OOFS = pd.concat(OOFS)
        score = metric_fn((OOFS['pred_correct'] > 0.63).astype(int), OOFS['correct'])
        print_log(f'OOF score: {score}')
        
if __name__ == '__main__':
    main()
