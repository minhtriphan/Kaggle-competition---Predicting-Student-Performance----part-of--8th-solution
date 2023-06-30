import torch
from torch.utils.data import Dataset
import pandas as pd

from custom_config import cfg, NUM_COLS, TXT_COLS, LEVEL2QUESTION

class PSPDataset(Dataset):
    def __init__(self, cfg, df, level = '0-4', df_labels = None):
        self.cfg = cfg
        self.level = level
        self.df_labels = df_labels
        data = df.groupby('session_id').apply(self._convert_dataframe_to_dict).to_frame()
        if df_labels is not None:
            # Main labels 
            main_data_labels = df_labels.loc[df_labels.qid.isin(LEVEL2QUESTION[level])]
            main_data_labels = main_data_labels.groupby('session').apply(lambda x: x['correct'].tolist()).to_frame().reset_index()
            main_data_labels.columns = ['session_id', 'main_correct']
            
            aux_data_labels = df_labels.loc[~df_labels.qid.isin(LEVEL2QUESTION[level])]
            aux_data_labels = aux_data_labels.groupby('session').apply(lambda x: x['correct'].tolist()).to_frame().reset_index()
            aux_data_labels.columns = ['session_id', 'aux_correct']
            
            # Add label
            self.data = data.merge(main_data_labels, left_index = True, right_on = 'session_id', how = 'inner').set_index('session_id')
            self.data = self.data.merge(aux_data_labels, left_index = True, right_on = 'session_id', how = 'inner').set_index('session_id')
        else:
            self.data = data
            self.data['main_correct'] = [[-1] * len(LEVEL2QUESTION[level])] * len(data)
            self.data['aux_correct'] = [[-1] * (18 - len(LEVEL2QUESTION[level]))] * len(data)
        
        # Extract the past information about the time to answer the questions
        if df_labels is not None:
            if level != '13-22':
                self.answering_time = df.groupby('session_id').apply(self._get_question_answering_time).loc[self.data.index].values
        self.data_index = self.data.index.tolist()
        self.main_label = self.data.main_correct.tolist()
        self.aux_label = self.data.aux_correct.tolist()
        self.data = self.data[0].tolist()
        
    def __len__(self):
        return len(self.data)
        
    def _convert_dataframe_to_dict(self, df):
        '''
        This function converts a dataframe to a dictionary. Currently, only convert numerical columns (NUM_COLS)
        '''
        data_dict = {}
        
        for col in NUM_COLS + TXT_COLS:
            data_dict[col] = df[col].tolist()
        return data_dict
    
    def _get_question_answering_time(self, df):
        if self.level == '0-4':
            return df.loc[(df.change_in_level == 1) & (df.level_group.isin(['5-12'])), 'time_diff'].sum() / (60 * 1e3)
        elif self.level == '5-12':
            return df.loc[(df.change_in_level == 1) & (df.level_group.isin(['13-22'])), 'time_diff'].sum() / (60 * 1e3)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        data_index = self.data_index[idx]
        data['session_id'] = data_index
        data['main_label'] = self.main_label[idx]
        data['aux_label'] = self.aux_label[idx]
        if self.df_labels is not None and self.level != '13-22':
            data['answering_time'] = self.answering_time[idx]
        else:
            data['answering_time'] = -1
        return data
    
class Collator(object):
    def __init__(self, cfg, level = '0-4'):
        self.cfg = cfg
        self.padding_idx = 0
        if level == '0-4':
            self.max_len = 512
        elif level == '5-12':
            self.max_len = 1024
        else:
            self.max_len = 1536
        
    def __call__(self, batch):
        # Remember to flip the sequence before padding or truncation
        inputs = {}
        
        for col in NUM_COLS + TXT_COLS:
            each_col_inputs = []
            mask = []
            for item in batch:
                seq = item[col]
                if len(seq) < self.max_len:
                    # Padding
                    item_mask = [0] * len(seq) + [1] * (self.max_len - len(seq))
                    seq = seq + [self.padding_idx] * (self.max_len - len(seq))
                else:
                    # Trimming
                    item_mask = [0] * self.max_len
                    seq = seq[-self.max_len:]    # Trim and get only the most recent activities
                each_col_inputs.append(seq)
                mask.append(item_mask)
            
            
            inputs[col] = torch.tensor(each_col_inputs, dtype = torch.long if col in TXT_COLS else torch.float)
            mask = torch.tensor(mask, dtype = torch.bool)
        
        # For labels
        inputs['session_id'] = torch.tensor([item['session_id'] for item in batch], dtype = torch.long)
        inputs['main_label'] = torch.tensor([item['main_label'] for item in batch], dtype = torch.float)
        inputs['aux_label'] = torch.tensor([item['aux_label'] for item in batch], dtype = torch.float)
        inputs['answering_time'] = torch.tensor([item['answering_time'] for item in batch], dtype = torch.float)
        
        return inputs, mask
