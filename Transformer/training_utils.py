import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# from tqdm import tqdm
from tqdm.notebook import tqdm

from custom_config import cfg, NUM_COLS, TXT_COLS, LEVEL2QUESTION
from kfold_split import split_data
from utils import metric_fn, print_log
from dataset import PSPDataset, Collator
from model import PSPModel

def train_epoch(cfg, model, train_dataloader, valid_dataloader, val_labels, optimizer, scheduler, epoch,
                level = '0-4', fold = 0, best_score = -np.inf):
    # Set up mix-precision training
    scaler = GradScaler(enabled = cfg.apex)
    
    loss = 0
    total_samples = 0
    global_step = 0
    
    val_schedule = [int(i) for i in list(np.linspace(1, len(train_dataloader), num = int(1 / cfg.val_check_interval) + 1, endpoint = True))[1:]]
    
    for i, (item, mask) in enumerate(train_dataloader):
        # Set up the training mode
        model.train()
        
        # Move the data into the device
        inputs = {k: v.to(cfg.device) for k, v in item.items() if k not in ['main_label', 'aux_label', 'session_id']}
        mask = mask.to(cfg.device)
        label = item['main_label'].to(cfg.device)
        aux_label = item['aux_label'].to(cfg.device)
        
        # Forward
        with autocast(enabled = cfg.apex):
            batch_loss, batch_output, _, _ = model(inputs, mask = mask, label = label, aux_label = aux_label)
            
        batch_size = batch_output.shape[0]
        
        # Backward
        scaler.scale(batch_loss).backward()
        
        # Update loss
        loss += batch_loss.item() * batch_size
        total_samples += batch_size
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        global_step += 1
        scheduler.step()
        
        if (i + 1) in val_schedule:
            print_log('Epoch: [{0}][{1}/{2}] - Start evaluating...'.format(epoch + 1, i + 1, len(train_dataloader)))
            val_score, val_loss, val_preds, val_trues, val_indexes = valid_epoch(cfg, model, valid_dataloader)
            oof = pd.DataFrame.from_dict(dict(zip(val_indexes, val_preds)), orient = 'index', columns = LEVEL2QUESTION[level])
            oof = organizing_oof(oof, level = level)
            oof = oof.merge(val_labels[['session_id', 'correct']], on = 'session_id', how = 'left')[['session_id', 'correct', 'pred_correct']]
            
            print_log('Train Loss: {train_loss:.4f} - '
                      'Val Score/Loss: {val_score:.4f}/{val_loss:.4f} - '
                      'LR: {lr:.8f}'
                      .format(train_loss = loss / total_samples,
                              val_score = val_score,
                              val_loss = val_loss,
                              lr = scheduler.get_lr()[0]))
            
            if val_score > best_score:
                # Store the model
                best_score = val_score
                print_log(f'Epoch [{epoch + 1}][{i + 1}/{len(train_dataloader)}] - The Best Score Updated to: {best_score:.4f}')
                # Store the model weights
                ckp = os.path.join(cfg.model_dir, cfg.ver[:-1], cfg.ver[-1], f"nn_level_{level.replace('-', '_')}_fold_{fold}.pt")
                torch.save(model.state_dict(), ckp)
                # Store the OOF
                oof.to_csv(os.path.join(cfg.model_dir, cfg.ver[:-1], cfg.ver[-1], f"oof_level_{level.replace('-', '_')}_fold_{fold}.csv"))
                
    return best_score
            
def valid_epoch(cfg, model, valid_dataloader):
    # Set up the evaluation mode
    model.eval()
    
    loss = 0
    total_samples = 0
    
    preds = []
    trues = []
    indexes = []
    
    for i, (item, mask) in enumerate(valid_dataloader):
        # Move the data into the device
        inputs = {k: v.to(cfg.device) for k, v in item.items() if k not in ['main_label', 'session_id']}
        mask = mask.to(cfg.device)
        label = item['main_label'].to(cfg.device)
        aux_label = item['aux_label'].to(cfg.device)
        index = item['session_id']
        batch_size = label.shape[0]
        
        with torch.no_grad():
            with autocast(enabled = cfg.apex):
                batch_loss, batch_output, _, _ = model(inputs, mask = mask, label = label, aux_label = aux_label)
                
                # Update loss
                loss += batch_loss.item() * batch_size
                total_samples += batch_size
                
        preds.append(batch_output.sigmoid().detach().cpu().numpy())
        trues.append(label.detach().cpu().numpy())
        indexes.append(index.numpy())
        
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    indexes = np.concatenate(indexes)
    
    score = metric_fn((preds > 0.63).astype(int), trues)
    
    return score, loss / total_samples, preds, trues, indexes

def embedding(cfg, model, dataloader, return_prediction = False):
    # Set up the evaluation mode
    model.eval()
    
    indexes = []
    predictions = []
    aux_predictions = []
    embeddings = []
    
    for i, (item, mask) in enumerate(dataloader):
        # Move the data into the device
        inputs = {k: v.to(cfg.device) for k, v in item.items() if k not in ['label', 'session_id']}
        mask = mask.to(cfg.device)
        index = item['session_id']
        batch_size = mask.shape[0]
        
        with torch.no_grad():
            with autocast(enabled = cfg.apex):
                _, batch_output, batch_aux_output, batch_features = model(inputs, mask = mask)
        
        indexes.append(index.numpy())
        predictions.append(batch_output.sigmoid().detach().cpu().numpy())
        aux_predictions.append(batch_aux_output.sigmoid().detach().cpu().numpy())
        embeddings.append(batch_features.detach().cpu().numpy())
        
    indexes = np.concatenate(indexes)
    predictions = np.hstack([np.concatenate(predictions), np.concatenate(aux_predictions)])
    embeddings = np.concatenate(embeddings)
    
    if return_prediction:
        return indexes, embeddings, predictions
    
    return indexes, embeddings, None

def get_optimizer(cfg, model):
    optimizer = AdamW(model.parameters(), lr = cfg.lr)
    return optimizer

def get_scheduler(cfg, optimizer, num_train_steps):
    return CosineAnnealingLR(optimizer, T_max = num_train_steps)

def fit(cfg, train, train_labels, TXT_COL_MAPS, fold = 0, level = '0-4'):
    # Split the data
    trn, val, trn_labels, val_labels = split_data(train, train_labels, fold = fold, level = level)
            
    # Show some statistics of the true labels
    print_log(f'The training/validation data size: {trn.shape[0]}/{val.shape[0]}')
    print_log(f'The training/validation data label size: {trn_labels.shape[0]}/{val_labels.shape[0]}')
    print_log('In the training set, the numbers of correct (1) and incorrect answers (0) are:')
    print_log(trn_labels['correct'].value_counts().to_dict())
    print_log('In the validation set, the numbers of correct (1) and incorrect answers (0) are:')
    print_log(val_labels['correct'].value_counts().to_dict())
    
    trn_dataset = PSPDataset(cfg, trn, level = level, df_labels = trn_labels)
    val_dataset = PSPDataset(cfg, val, level = level, df_labels = val_labels)
    
    if level == '0-4':
        batch_size = 256
    elif level == '5-12':
        batch_size = 128
    else:
        batch_size = 64
    
    trn_dataloader = DataLoader(trn_dataset, batch_size = batch_size, num_workers = cfg.num_workers, collate_fn = Collator(cfg, level), shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size * 2, num_workers = cfg.num_workers, collate_fn = Collator(cfg, level), shuffle = False)
    
    model = PSPModel(cfg, TXT_COL_MAPS, level = level).to(cfg.device)
    optimizer = get_optimizer(cfg, model)
    num_train_steps = len(trn_dataloader) * cfg.nepochs
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)
    
    best_score = -np.inf
    if cfg.use_tqdm:
        tbar = tqdm(range(cfg.nepochs))
    else:
        tbar = range(cfg.nepochs)
    for epoch in tbar:
        best_score = train_epoch(cfg, model, trn_dataloader, val_dataloader, val_labels, optimizer, scheduler, epoch,
                                 level = level, fold = fold, best_score = best_score)
    return best_score

def evaluate(cfg, train, train_labels, TXT_COL_MAPS, fold = 0, level = '0-4'):
    # Split the data
    trn, val, trn_labels, val_labels = split_data(train, train_labels, fold = fold, level = level)
            
    # Show some statistics of the true labels
    print_log(f'The training/validation data size: {trn.shape[0]}/{val.shape[0]}')
    print_log(f'The training/validation data label size: {trn_labels.shape[0]}/{val_labels.shape[0]}')
    print_log('In the training set, the numbers of correct (1) and incorrect answers (0) are:')
    print_log(trn_labels['correct'].value_counts().to_dict())
    print_log('In the validation set, the numbers of correct (1) and incorrect answers (0) are:')
    print_log(val_labels['correct'].value_counts().to_dict())
    
    if level == '0-4':
        batch_size = 256
    elif level == '5-12':
        batch_size = 128
    else:
        batch_size = 64
    
    val_dataset = PSPDataset(cfg, val, level = level, df_labels = val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size * 2, num_workers = cfg.num_workers, collate_fn = Collator(cfg, level))
    
    model = PSPModel(cfg, TXT_COL_MAPS, level = level).to(cfg.device)
    ckp = torch.load(os.path.join(cfg.model_dir, cfg.ver[:-1], cfg.ver[-1], f"nn_level_{level.replace('-', '_')}_fold_{fold}.pt"), map_location = cfg.device)
    model.load_state_dict(ckp)
    
    score, loss, preds, trues, indexes = valid_epoch(cfg, model, val_dataloader)
    
    oof = pd.DataFrame.from_dict(dict(zip(indexes, preds)), orient = 'index', columns = LEVEL2QUESTION[level])
    
    oof = organizing_oof(oof, level = level)
    oof = oof.merge(val_labels[['session_id', 'correct']], on = 'session_id', how = 'left')[['session_id', 'correct', 'pred_correct']]
    
    score = metric_fn((oof['pred_correct'] > 0.63).astype(int), oof['correct'])
    
    return score, oof

def infer_embedding(cfg, train, train_labels, TXT_COL_MAPS, fold = 0, level = '0-4', return_prediction = False):
    # Split the data
    trn, val, trn_labels, val_labels = split_data(train, train_labels, fold = fold, level = level)
    
    trn_dataset = PSPDataset(cfg, trn, level = level, df_labels = trn_labels)
    val_dataset = PSPDataset(cfg, val, level = level, df_labels = val_labels)
    
    if level == '0-4':
        batch_size = 256
    elif level == '5-12':
        batch_size = 128
    else:
        batch_size = 64
        
    trn_dataloader = DataLoader(trn_dataset, batch_size = batch_size, num_workers = cfg.num_workers, collate_fn = Collator(cfg, level))
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size * 2, num_workers = cfg.num_workers, collate_fn = Collator(cfg, level))
    
    model = PSPModel(cfg, TXT_COL_MAPS, level = level).to(cfg.device)
    ckp = torch.load(os.path.join(cfg.model_dir, cfg.ver[:-1], cfg.ver[-1], f"nn_level_{level.replace('-', '_')}_fold_{fold}.pt"), map_location = cfg.device)
    model.load_state_dict(ckp)
    
    trn_indexes, trn_embeddings, trn_predictions = embedding(cfg, model, trn_dataloader, return_prediction = return_prediction)
    val_indexes, val_embeddings, val_predictions = embedding(cfg, model, val_dataloader, return_prediction = return_prediction)
    
    return trn_indexes, trn_embeddings, trn_predictions, val_indexes, val_embeddings, val_predictions
    
def organizing_oof(oof, level = '0-4'):
    transformed_oof = {}
    
    for idx, row in oof.iterrows():
        transformed_oof[idx] = [row.tolist()]
    
    transformed_oof = pd.DataFrame.from_dict(transformed_oof, orient = 'index')
    transformed_oof = transformed_oof.explode(0).reset_index()
    transformed_oof.columns = ['session', 'pred_correct']
    transformed_oof['qid'] = transformed_oof.groupby('session').agg('cumcount') + int(LEVEL2QUESTION[level][0][1:])
    transformed_oof['qid'] = transformed_oof['qid'].apply(lambda x: f'q{int(x)}')
    transformed_oof['session_id'] = transformed_oof[['session', 'qid']].apply(lambda x: f"{x['session']}_{x['qid']}", axis = 1)
    return transformed_oof
