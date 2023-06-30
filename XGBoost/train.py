# Python packages
import os, pickle
import numpy as np
import pandas as pd
import polars as pl
    
# Custom packages
from custom_config import cfg, set_random_seed, LEVEL2QUESTION
from kfold_split import fold_assign, split_data
from feature_engineering import process_data, add_columns_pl, feature_engineering_pl
from utils import build_log, print_log, metric_fn
from model import DEFAULT_LGBM_PARAMS, DEFAULT_XGB_PARAMS, BoostingModel

import warnings
warnings.filterwarnings('ignore')

def main():
    ################ Initial settings ################
    # Set seed
    set_random_seed(cfg.seed)
    
    # Build log
    if cfg.use_log:
        build_log(cfg)
        
    oofs = []
        
    for fold in cfg.training_folds:
        print_log(f' FOLD {fold} '.center(50, '*'))
        
        ################ Import and process the data ################
        if cfg.use_polar:
            train = pl.read_csv(os.path.join(cfg.comp_data_dir, 'train.csv'))
        else:
            train = pd.read_csv(os.path.join(cfg.comp_data_dir, 'train.csv'))
        train_labels = pd.read_csv(os.path.join(cfg.comp_data_dir, 'train_labels.csv'))
        
        train = process_data(train, only_drop = True)
        
        # Process train labels
        train_labels['session'] = train_labels['session_id'].apply(lambda x: x.split('_')[0]).astype(np.int64)
        train_labels['qid'] = train_labels['session_id'].apply(lambda x: x.split('_')[1])
            
        ################ KFold split ################
        if not cfg.done_kfold_split:
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
            
            if cfg.use_polar:
                train = train.to_pandas()
            
            train = train.merge(session2fold, on = 'session_id', how = 'inner')
            train_labels = train_labels.merge(session2fold, left_on = 'session', right_on = 'session_id', 
                                              how = 'left', suffixes = ('', '_')).drop(['session_id_'], axis = 1)
            
            if cfg.use_polar:
                train = pl.from_pandas(train)
            
        # Add new columns
        new_columns = add_columns_pl(train)
        train = train.with_columns(new_columns)
    
        all_sessions = train_labels.loc[train_labels.fold == fold, 'session'].unique().tolist()
    
        oof = pd.DataFrame(data = np.zeros((len(all_sessions), 18)), index = all_sessions)
        
        CACHE_DATA = {}
        
        for i, level in enumerate(['0-4', '5-12', '13-22']):
            print_log(f' Level {level} '.center(50, '-'))
            trn, val, trn_labels, val_labels = split_data(train, train_labels, fold = fold, level = level)
            
            # Store the feature list
            with open(os.path.join(cfg.model_dir, cfg.ver[:-1], cfg.ver[-1], f"feature_list_level_{level.replace('-', '_')}.pkl"), 'rb') as f:
                remaining_features = pickle.load(f)
            
            # Feature engineering
            agg_trn, _ = feature_engineering_pl(trn, group = level, use_extra_features = True, feature_suffix = '', remaining_features = remaining_features)
            agg_val, _ = feature_engineering_pl(val, group = level, use_extra_features = True, feature_suffix = '', remaining_features = remaining_features)
            
            agg_trn = agg_trn[remaining_features]
            agg_val = agg_val[remaining_features]

            assert agg_trn.shape[1] == agg_val.shape[1]
            assert agg_trn.session_id.nunique() == trn.select(pl.col('session_id').n_unique()).item()

            agg_trn['session_id'] = agg_trn['session_id'].astype('int64')
            agg_val['session_id'] = agg_val['session_id'].astype('int64')
            
            agg_trn = agg_trn.set_index('session_id')
            agg_val = agg_val.set_index('session_id')
            
            print_log(f'Level {level}, number of initial features is {len(remaining_features)}')
            
            # Add the features of the previous level group
            current_features = [j + f'_{i}' for j in agg_trn.columns]
            rename_columns = dict(zip(agg_trn.columns, current_features))
            agg_trn = agg_trn.rename(columns = rename_columns)
            agg_val = agg_val.rename(columns = rename_columns)
                
            if level == '5-12':
                past_levels = ['0-4']
                added_features = 0
                for past_level in past_levels:
                    past_agg_trn, past_agg_val = CACHE_DATA[past_level]
                    agg_trn = agg_trn.merge(past_agg_trn, left_index = True, right_index = True)
                    agg_val = agg_val.merge(past_agg_val, left_index = True, right_index = True)
                    added_features += past_agg_trn.shape[1]
                    
            elif level == '13-22':
                past_levels = ['0-4', '5-12']
                added_features = 0
                for past_level in past_levels:
                    past_agg_trn, past_agg_val = CACHE_DATA[past_level]
                    agg_trn = agg_trn.merge(past_agg_trn, left_index = True, right_index = True)
                    agg_val = agg_val.merge(past_agg_val, left_index = True, right_index = True)
                    added_features += past_agg_trn.shape[1]
                    
            agg_trn = agg_trn.reset_index()
            agg_val = agg_val.reset_index()
            
            # Add target columns
            for q in LEVEL2QUESTION[level]:
                sub_trn_labels = trn_labels.loc[trn_labels.qid == q]
                sub_val_labels = val_labels.loc[val_labels.qid == q]
                
                agg_trn = agg_trn.merge(sub_trn_labels[['session', 'correct']], left_on = 'session_id', right_on = 'session', how = 'left').drop('session', axis = 1)
                agg_val = agg_val.merge(sub_val_labels[['session', 'correct']], left_on = 'session_id', right_on = 'session', how = 'left').drop('session', axis = 1)
                
                agg_trn = agg_trn.rename(columns = {'correct': f'correct_{q}'})
                agg_val = agg_val.rename(columns = {'correct': f'correct_{q}'})
            
            # Add the embedding
            embedding_path = os.path.join(os.path.join(cfg.model_dir, cfg.embedding_ver[:-1], cfg.embedding_ver[-1]))
            embedding_trn = pd.read_csv(os.path.join(embedding_path, f"{cfg.embedding_ver}_trn_embeddings_level_{level.replace('-', '_')}_fold_{fold}.csv"), index_col = 0)
            embedding_val = pd.read_csv(os.path.join(embedding_path, f"{cfg.embedding_ver}_val_embeddings_level_{level.replace('-', '_')}_fold_{fold}.csv"), index_col = 0)
            embedding_trn.columns = [f'{j}_{i}' for j in embedding_trn.columns]
            embedding_val.columns = [f'{j}_{i}' for j in embedding_val.columns]
            
            agg_trn = pd.concat([agg_trn, embedding_trn.loc[agg_trn.session_id].reset_index(drop = True)], axis = 1)
            agg_val = pd.concat([agg_val, embedding_val.loc[agg_val.session_id].reset_index(drop = True)], axis = 1)
            
            # Initialize the model
            feature_cols = [c for c in agg_trn.columns if c not in ['session_id'] + [f'correct_{q}' for q in LEVEL2QUESTION[level]]]
            target_col = [f'correct_{q}' for q in LEVEL2QUESTION[level]]
            
            # Store the feature list
            with open(os.path.join(cfg.model_dir, cfg.ver[:-1], cfg.ver[-1], f"feature_cols_level_{level.replace('-', '_')}_fold_{fold}.pkl"), 'wb') as f:
                pickle.dump(feature_cols, f)
            
            model = BoostingModel(cfg, 
                                  'xgb', 
                                  DEFAULT_XGB_PARAMS, 
                                  agg_trn, 
                                  feature_cols = feature_cols, 
                                  target_col = target_col, 
                                  eval_data = agg_val,
                                  level = level,
                                  fold = fold)
            
            model.fit()
            valid_score, q_pred = model.evaluate()
            start_index = int(LEVEL2QUESTION[level][0][1:]) - 1
            end_index = start_index + q_pred.shape[1]
            oof.loc[agg_val.session_id, list(range(start_index,end_index))] = q_pred
            print_log(f'Level {level}: {valid_score}')
            
            # Add predicted answers from the previous levels
            pred_train = model.predict(agg_trn)
            agg_trn[[f'pred_{i}' for i in LEVEL2QUESTION[level]]] = pred_train
            agg_val[[f'pred_{i}' for i in LEVEL2QUESTION[level]]] = q_pred
            
            # Don't forget to drop the target columns
            agg_trn = agg_trn.drop(target_col, axis = 1)
            agg_val = agg_val.drop(target_col, axis = 1)
            
            CACHE_DATA[level] = (agg_trn.set_index('session_id'), agg_val.set_index('session_id'))
        
        # Re-organize the OOF file and compute the CV
        transformed_oof = organizing_oof(oof)
        val_labels = train_labels.loc[train_labels.fold == fold]
        oof = val_labels.merge(transformed_oof[['session_id', 'pred_correct']], on = 'session_id', how = 'inner')
        oof = oof[['session_id', 'correct', 'pred_correct']]
        score = metric_fn((oof['pred_correct'] > 0.63).astype(int), oof['correct'])
        print_log(f'Fold {fold}: {score}')
        oof.to_csv(os.path.join(cfg.model_dir, cfg.ver[:-1], cfg.ver[-1], f'{cfg.ver}_oofs_fold_{fold}.csv'), index = False)
        
        oofs.append(oof)
        
    oofs = pd.concat(oofs)
    oof_score = metric_fn((oofs['pred_correct'] > 0.63).astype(int), oofs['correct'])
    print_log('*' * 50)
    print_log(f'OOF score: {oof_score}')
            
def organizing_oof(oof):
    transformed_oof = {}
    
    for idx, row in oof.iterrows():
        transformed_oof[idx] = [row.tolist()]
    
    transformed_oof = pd.DataFrame.from_dict(transformed_oof, orient = 'index')
    transformed_oof = transformed_oof.explode(0).reset_index()
    transformed_oof.columns = ['session', 'pred_correct']
    transformed_oof['qid'] = transformed_oof.groupby('session').agg('cumcount')
    transformed_oof['qid'] = transformed_oof['qid'].apply(lambda x: f'q{int(x) + 1}')
    transformed_oof['session_id'] = transformed_oof[['session', 'qid']].apply(lambda x: f"{x['session']}_{x['qid']}", axis = 1)
    return transformed_oof
        
if __name__ == '__main__':
    main()
