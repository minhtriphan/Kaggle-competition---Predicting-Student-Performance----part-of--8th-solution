import os
import random
import torch
import numpy as np

class cfg:
    # General settings
    comp_name = 'PSP'
    ver = 'v8a'
    env = 'paperspace'    # 'paperspace', 'vastai', 'kaggle', 'colab'
    if env == 'colab':
        from google.colab import drive
        drive.mount('/content/drive')
    seed = 314
    mode = 'train'    # 'train', 'eval', 'infer_embedding'
    use_tqdm = True
    use_log = False
    use_polar = False    # If False, use pandas instead
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Data
    nfolds = 5
    training_folds = [0, 1, 2, 3, 4]
    done_kfold_split = True
    # For dataset and dataloaders
    max_len = 512
    batch_size = 64
    num_workers = os.cpu_count()
    # For training and validating
    apex = True
    nepochs = 50
    val_check_interval = 0.5    # Evaluate after each epoch
    # For the optimizer and the scheduler
    lr = 1e-3
    weight_decay = 1e-2
    # Paths
    if env == 'colab':
        comp_data_dir = f'/content/drive/My Drive/Kaggle competitions/{comp_name}/comp_data'
        ext_data_dir = f'/content/drive/My Drive/Kaggle competitions/{comp_name}/ext_data'
        model_dir = f'/content/drive/My Drive/Kaggle competitions/{comp_name}/model'
    elif env == 'kaggle':
        comp_data_dir = '/kaggle/input/predict-student-performance-from-game-play'
        ext_data_dir = ...
        model_dir = '.'
    elif env in ['paperspace', 'vastai']:
        comp_data_dir = 'data'
        ext_data_dir = 'ext_data'
        model_dir = 'model'
    os.makedirs(os.path.join(model_dir, ver[:-1], ver[-1]), exist_ok = True)

def set_random_seed(seed, use_cuda = True):
    np.random.seed(seed) # cpu vars
    torch.manual_seed(seed) # cpu  vars
    random.seed(seed) # Python
    os.environ['PYTHONHASHSEED'] = str(seed) # Python hash building
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

NUM_COLS = ['index', 'time_diff', 'room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration']
TXT_COLS = ['level', 'event_name', 'name', 'text', 'fqid', 'room_fqid', 'text_fqid']

QUESTION2LEVEL = {}    # This dictionary contains pairs of {k: v}, which means, to predict question k, we need data from group v
    
for i in range(1, 19):
    # From https://www.kaggle.com/code/cdeotte/xgboost-baseline-0-676
    # There are 18 questions from 1 to 18
    if i <= 3:
        QUESTION2LEVEL[f'q{i}'] = '0-4'
    elif i <= 13:
        QUESTION2LEVEL[f'q{i}'] = '5-12'
    else:
        QUESTION2LEVEL[f'q{i}'] = '13-22'
        
LEVEL2QUESTION = {
    '0-4': ['q1', 'q2', 'q3'],
    '5-12': ['q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13'],
    '13-22': ['q14', 'q15', 'q16', 'q17', 'q18'],
}

LEVEL_MAP = {
        '0-4': 0,
        '5-12': 1,
        '13-22': 2,
    }
