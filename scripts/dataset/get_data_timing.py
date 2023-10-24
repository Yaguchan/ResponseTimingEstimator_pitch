import os
import json
import torch
import random
import argparse
import numpy as np
from dotmap import DotMap
from tqdm import tqdm

from src.utils.utils import load_config

# sakuma
from src.datasets.dataset_timing import get_dataloader, get_dataset
# python scripts/dataset/get_data_timing.py configs/dataset/config.json --gpuid 0 --cv_id 1

# annotated
# from src.datasets.dataset_annotated_timing import get_dataloader, get_dataset
# python scripts/dataset/get_data_timing.py configs/dataset/annotated_config.json --gpuid 0 --cv_id 1

SPEAKERS = ['M1']
FILES = ['M1_all']
TRAIN_FILES = ['M1_train']
VALID_FILES = ['M1_train']
TEST_FILES = ['M1_test']

OUTDIR = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/scripts/dataset/text.txt'


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def get_text(loader, split):
    list_data = []
    list_wav = []
    timing = [0 for i in range(11)]
    len_text = [0 for i in range(21)]
    
    for batch in tqdm(loader):
        for i in range(len(batch[0])):
            chs = batch[0][i]
            texts = batch[1][i]
            kanas = batch[2][i]
            idxs = batch[3][i]
            vad = batch[4][i]
            turn = batch[5][i]
            last_ipu = batch[6][i]
            targets = batch[7][i]
            feats = batch[8][i]
            input_lengths = batch[9][i]
            offsets = batch[10][i]
            indices = batch[11][i]
            is_barge_in = batch[12][i]
            names = batch[13][i]
            # das = batch[14][i]
            # da = batch[15][i]
            # nxt_das = batch[16][i]
            # nxt_da = batch[17][i]
            # wav_path = batch[18][i]
            
            
            # if offsets >= 1800: list_wav.append(f'{wav_path}: {offsets}: {texts[-1]}')
            # list_data.append(len(texts[-1]))

            # 発話タイミング分布
            if offsets < -250: timing[0] += 1
            elif -250 <= offsets < 0: timing[1] += 1
            elif 0 <= offsets < 250: timing[2] += 1
            elif 250 <= offsets < 500: timing[3] += 1
            elif 500 <= offsets < 750: timing[4] += 1
            elif 750 <= offsets < 1000: timing[5] += 1
            elif 1000 <= offsets < 1250: timing[6] += 1
            elif 1250 <= offsets < 1500: timing[7] += 1
            elif 1500 <= offsets < 1750: timing[8] += 1
            elif 1750 <= offsets <= 2000: timing[9] += 1
            else: timing[10] += 1
            
            # ユーザの発話長
            if len(texts[-1]) < 60:
                len_text[len(texts[-1])//3] += 1
            else:
                len_text[20] += 1
            """
            if len(texts[-1]) < 3:
                list_wav.append(wav_path)
            """
    
    """
    list_wav.sort()
    for x in list_wav:
        print(x)
    print()
    """
    
    print('timing:', timing)
    print('len_text:', len_text)
    
    """
    with open(OUTDIR, mode='a') as f:
        f.write(split + "\n")
        for data in list_data:
            f.write(data + "\n")
    """
    return len_text


def cross_validation(config, device, cv_id, out_name):        
    train_dataset = get_dataset(config, cv_id, split='train', subsets=FILES, speaker_list=SPEAKERS)
    val_dataset = get_dataset(config, cv_id, split='test', subsets=FILES, speaker_list=SPEAKERS)   
    train_loader = get_dataloader(train_dataset, config, shuffle=True)
    val_loader = get_dataloader(val_dataset, config, shuffle=False)
    del train_dataset
    del val_dataset
    lenx = get_text(train_loader, 'train')
    leny = get_text(val_loader, 'valid')
    lenz = [x+y for x, y in zip(lenx, leny)]
    for z in lenz: print(z)


def run(config, device, cv_id, out_name):
    train_dataset = get_dataset(config, cv_id, split='train', subsets=FILES, speaker_list=SPEAKERS)
    val_dataset = get_dataset(config, cv_id, split='valid', subsets=FILES, speaker_list=SPEAKERS)
    test_dataset = get_dataset(config, cv_id, split='test', subsets=FILES, speaker_list=SPEAKERS) 
    train_loader = get_dataloader(train_dataset, config, shuffle=True)
    val_loader = get_dataloader(val_dataset, config, shuffle=False)
    test_loader = get_dataloader(test_dataset, config, shuffle=False)
    loader_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    del train_dataset
    del val_dataset
    del test_dataset
    
    

def set_config_device(args):
    config = load_config(args.config)
    seed_everything(config.seed)
    
    if args.gpuid >= 0:
        config.gpu_device = args.gpuid
    
    if config.cuda:
        device = torch.device('cuda:{}'.format(config.gpu_device))
    else:
        device = torch.device('cpu')
    
    # cross validation
    # config.model_params.lm_model_path = os.path.join(config.model_params.lm_model_path, 'cv{}'.format(str(args.cv_id)), 'best_val_bacc_model.pth')
    cross_validation(config, device, int(args.cv_id), 'cv{}'.format(int(args.cv_id)))
    
    # all data
    # config.model_params.lm_model_path = os.path.join(config.model_params.lm_model_path, 'M1_all', 'best_val_bacc_model.pth')
    # run(config, device, int(args.cv_id), 'M1_all')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs/config.path', help='path to config file')
    parser.add_argument('--gpuid', type=int, default=-1, help='gpu device id')
    parser.add_argument('--cv_id', type=int, default=0, help='id for the cross validation settings')
    args = parser.parse_args()
    set_config_device(args)