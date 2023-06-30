
import os
import random
import torch
import numpy as np

class cfg:
    # General settings
    comp_name = 'PSP'
    ver = 'v7b'
    embedding_ver = 'v5c'
    env = 'vastai'    # 'paperspace', 'vastai', 'kaggle', 'colab'
    if env == 'colab':
        from google.colab import drive
        drive.mount('/content/drive')
    seed = 314
    use_tqdm = True
    use_log = False
    use_polar = True    # If False, use pandas instead
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Data and training
    nfolds = 5
    training_folds = [0, 1, 2, 3, 4]
    done_kfold_split = True
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
