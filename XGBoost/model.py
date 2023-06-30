import os, pickle, math
import numpy as np
import torch
import lightgbm as lgbm
import xgboost as xgb
import warnings
from typing import List, Optional, Union, Tuple
from scipy.special import expit

from custom_config import cfg
from utils import print_log, metric_fn

DEFAULT_LGBM_PARAMS = {
    'boosting_type': 'gbdt',
    'max_depth': 4,
    # 'objective': 'binary',
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'learning_rate': 0.05,
    'num_leaves': 5,
    'random_state': cfg.seed,
    'device_type': 'cpu',
    'gpu_platform_id': None,
    'gpu_device_id': None,
    'force_col_wise': 'true',
}

DEFAULT_XGB_PARAMS = {
    'booster': 'gbtree',
    'max_depth': 4,
    'objective': 'binary:logistic',
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'learning_rate': 0.05,
    'random_state': cfg.seed,
    'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
}

class MultiLabelDatasetForLGBM(lgbm.Dataset):
    '''
    Makeshift Class for storing multi label.
    label: numpy.ndarray (n_example, n_target)
    Adopted from: https://www.kaggle.com/code/ttahara/custom-lgbm-for-multi-class-multi-task
    '''
    def __init__(self, data, label = None, reference = None, weight = None, group = None, init_score = None, silent = False,
        feature_name = 'auto', categorical_feature = 'auto', params = None, free_raw_data = True):
        '''Initialize'''
        if label is not None:
            # Make dummy 1D-array
            dummy_label = np.arange(len(data))
            
        super(MultiLabelDatasetForLGBM, self).__init__(
            data, dummy_label, reference, weight, group, init_score, silent,
            feature_name, categorical_feature, params, free_raw_data)
        
        self.multi_label = label
        
    def get_multi_label(self):
        '''Get 2D-array label'''
        return self.multi_label
    
    def set_multi_label(self, multi_label: np.ndarray):
        '''Set 2D-array label'''
        self.multi_label = multi_label
        return self

class MultiLabelBCELossLGBM(object):
    '''
    This object is the implementation of the multi-label classification for LGBM. It uses the Binary Cross Entropy loss.
    Adapted from: https://www.kaggle.com/code/ttahara/custom-lgbm-for-multi-class-multi-task
    '''
    def __init__(self, n_class: int = 3, epsilon: float = 1e-32):
        self.n_class = n_class
        self.epsilon = epsilon
        self.name = 'multi-label loss'
        
    def _get_prob_value(self, preds: np.ndarray) -> np.ndarray:
        '''Convert Logit to Prob by Sigmoid.'''
        prob = np.clip(expit(preds), self.epsilon, 1 - self. epsilon)
        return prob
        
    def __call__(self, preds: np.ndarray, true: np.ndarray, weight: Optional[np.ndarray] = None) -> float:
        prob = self._get_prob_value(preds)
        per_sample_loss = np.mean(-(true * np.log(prob) + (1 - true) * np.log(1 - prob)), axis = 1)
        return np.average(per_sample_loss, weight)
    
    def _calc_grad_and_hess(self, preds: np.ndarray, true: np.ndarray, weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray]:
        prob = self._get_prob_value(preds)
        gradient = prob - true
        hessian = prob * (1 - prob)
        
        if weight is not None:
            gradient = gradient * weight[:, None]
            hessian = hessian * weight[:, None]
            
        return gradient, hessian
    
    def return_loss(self, preds: np.ndarray, data: lgbm.Dataset) -> Tuple[str, float, bool]:
        '''Return Loss for lightgbm'''
        true = data.get_multi_label()
        weight = data.get_weight()
        n_example = len(true)
        
        # Reshape preds: (n_class * n_example,) -> (n_class, n_example) ->  (n_example, n_class)
        preds = preds.reshape(self.n_class, n_example).T  # Convert the initial predictions from a 1D-array to a 2D-array
        
        # Compute loss
        loss = self(preds, true, weight)
        
        return self.name, loss, False
    
    def return_grad_and_hess(self, preds: np.ndarray, data: lgbm.Dataset) -> Tuple[np.ndarray]:
        """Return Grad and Hess for lightgbm"""
        true = data.get_multi_label()
        weight = data.get_weight()
        n_example = len(true)
        
        # Reshape preds: (n_class * n_example,) -> (n_class, n_example) ->  (n_example, n_class)
        preds = preds.reshape(self.n_class, n_example).T  # Convert the initial predictions from a 1D-array to a 2D-array
        
        # Compute the gradient and hessian
        gradient, hessian =  self._calc_grad_and_hess(preds, true, weight)
        
        # The gradient and hessian are now 2D-arrays (n_class, n_example), we convert them back to the 1D-arrays (n_class * n_example,)
        gradient = gradient.T.reshape(n_example * self.n_class)
        hessian = hessian.T.reshape(n_example * self.n_class)
        
        return gradient, hessian

class BoostingModel():
    def __init__(
        self,
        cfg,
        model,    # Either 'xgb' or 'lgbm'
        model_params,
        data,
        feature_cols,
        target_col = 'correct',
        eval_data = None,
        level = '0-4',
        fold = 0
    ):
        self.cfg  = cfg
        self.model = model
        assert model in ['xgb', 'lgbm'], "The current version only supports 'xgb' (XGBoost) or 'lgbm' (LightGBM)."
        self.model_params = model_params
        if model == 'lgbm':
            warnings.warn('The current version only supports training LightGBM with CPU')
            self.multilabelloss = MultiLabelBCELossLGBM(n_class = len(target_col))
            self.model_params['num_class'] = len(target_col)
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.eval_data = eval_data
        self.level = level
        self.fold = fold

        if cfg.env == 'kaggle':
            self.model_dir = os.path.join(cfg.model_dir, 'best_model')
        else:
            self.model_dir = os.path.join(cfg.model_dir, cfg.ver[:-1], cfg.ver[-1])
    
    # Prepare the data for training and validation
    def _prepare_data_lgbm_train(self):
        if len(self.target_col) == 1:
            trn_ds = lgbm.Dataset(self.data[self.feature_cols].values, label = self.data[self.target_col].values)
        else:
            trn_ds = MultiLabelDatasetForLGBM(self.data[self.feature_cols].values, label = self.data[self.target_col].values)
        if self.eval_data is not None:
            if len(self.target_col) == 1:
                val_ds = lgbm.Dataset(self.eval_data[self.feature_cols].values, label = self.eval_data[self.target_col].values)
            else:
                val_ds = MultiLabelDatasetForLGBM(self.eval_data[self.feature_cols].values, label = self.eval_data[self.target_col].values)
            
            return trn_ds, val_ds
        return trn_ds
    
    def _prepare_data_xgb_train(self, mode = 'train'):
        trn_ds = xgb.DMatrix(self.data[self.feature_cols], label = self.data[self.target_col])
        if self.eval_data is not None:
            val_ds = xgb.DMatrix(self.eval_data[self.feature_cols], label = self.eval_data[self.target_col])
            
            return trn_ds, val_ds
        return trn_ds
    
    def _prepare_data_train(self):
        if self.model == 'lgbm':
            datasets = self._prepare_data_lgbm_train()
        elif self.model == 'xgb':
            datasets = self._prepare_data_xgb_train()
        return datasets
    
    # Prepare the data for predicting
    def _prepare_data_predict(self, data):
        if self.model == 'lgbm':
            pred_ds = data[self.feature_cols].values
        elif self.model == 'xgb':
            pred_ds = xgb.DMatrix(data[self.feature_cols])
        return pred_ds
    
    # Training functions
    def _train_lgbm(self):
        datasets = self._prepare_data_train()
        if self.eval_data is not None:
            train_set, eval_set = datasets
            valid_sets = [eval_set]
            valid_names = ['valid']
            callbacks = [lgbm.early_stopping(100), lgbm.log_evaluation(100)]
        else:
            train_set = datasets
            valid_sets = None
            valid_names = None
            callbacks = None
        
        # Custom metric for LGBM
        def custom_metric_lgbm(
            preds: np.ndarray, data: lgbm.Dataset, threshold: float = 0.63,
        ) -> Tuple[str, float, bool]:
            '''Calculate Custom F1-score'''
            if len(self.target_col) > 1:
                label = data.get_multi_label()
                n_example = len(label)
                preds = preds.reshape(len(self.target_col), n_example).T
                f1_score = metric_fn((expit(preds) > threshold).astype(int), label)
            else:
                label = data.get_label()
                f1_score = metric_fn((preds > threshold).astype(int), label)
            
            return 'custom_f1_score', f1_score, True    # eval_name, eval_result, is_higher_better
        
        print_log('Start training...')
        model = lgbm.train(
            self.model_params,
            train_set = train_set,
            valid_sets = valid_sets,
            valid_names = valid_names,
            num_boost_round = 1000,
            callbacks = callbacks,
            fobj = self.multilabelloss.return_grad_and_hess,
            feval = lambda preds, data: [self.multilabelloss.return_loss(preds, data),
                                         custom_metric_lgbm(preds, data)],
        )
        
        # Model checkpoint
        ckp = os.path.join(self.model_dir, f"{self.model}_level_{self.level.replace('-', '_')}_fold_{self.fold}.txt")
        print_log(f'Saving the model to {ckp}...')
        model.save_model(ckp, num_iteration = model.best_iteration)
    
    def _train_xgb(self):
        datasets = self._prepare_data_train()
        if self.eval_data is not None:
            d_train, d_valid = datasets
            evals = [(d_train, 'train'), (d_valid, 'valid')]
        else:
            d_train = datasets
            evals = None
            
        # Custom metric for XGB
        def custom_metric_xgb(
            preds: np.ndarray, data: xgb.DMatrix, threshold: float = 0.63,
        ) -> Tuple[str, float]:
            '''Calculate Custom F1-score'''
            label = data.get_label()
            f1_score = metric_fn((preds > threshold).astype(int), label)
            
            return 'custom_f1_score', f1_score    # eval_name, eval_result
            
        print_log('Start training...')
        model = xgb.train(
            self.model_params,
            dtrain = d_train,
            evals = evals,
            num_boost_round = 1000,
            verbose_eval = 100,
            early_stopping_rounds = 100,
            custom_metric = custom_metric_xgb,
            maximize = True,
        )
        
        # Model checkpoint
        ckp = os.path.join(self.model_dir, f"{self.model}_level_{self.level.replace('-', '_')}_fold_{self.fold}.pkl")
        print_log(f'Saving model to {ckp}...')
        with open(ckp, 'wb') as f:
            pickle.dump(model, f)
        
    def fit(self):
        if self.model == 'lgbm':
            self._train_lgbm()
        elif self.model == 'xgb':
            self._train_xgb()
            
    # Evaluation
    def evaluate(self):
        print_log('Start evaluating...')
        y_pred = self.predict(self.eval_data, return_proba = True)
        y_true = self.eval_data[self.target_col].values
        return metric_fn((y_pred > 0.63).astype(int), y_true), y_pred
    
    # Predicting functions
    def load_model(self):
        if self.model == 'lgbm':
            ckp = os.path.join(self.model_dir, f"{self.model}_level_{self.level.replace('-', '_')}_fold_{self.fold}.txt")
            print_log(f'Loading the model from {ckp}...')
            model = lgbm.Booster(model_file = ckp)
        elif self.model == 'xgb':
            ckp = os.path.join(self.model_dir, f"{self.model}_level_{self.level.replace('-', '_')}_fold_{self.fold}.pkl")
            print_log(f'Loading the model from {ckp}...')
            with open(ckp, 'rb') as f:
                model = pickle.load(f)
        return model
    
    def _predict_lgbm(self, data):
        pred_ds = self._prepare_data_predict(data)
        model = self.load_model()
        
        prediction = model.predict(pred_ds)
        if len(self.target_col) > 1:
            prediction = expit(prediction)
        return prediction
    
    def _predict_xgb(self, data):
        pred_ds = self._prepare_data_predict(data)
        model = self.load_model()
        
        prediction = model.predict(pred_ds, iteration_range = (0, model.best_iteration + 1))
        return prediction
    
    def predict(self, data, return_proba = True):
        if self.model == 'lgbm':
            prediction = self._predict_lgbm(data)
        elif self.model == 'xgb':
            prediction = self._predict_xgb(data)
        
        if not return_proba:
            prediction = (prediction > 0.63).astype(int)
        return prediction
