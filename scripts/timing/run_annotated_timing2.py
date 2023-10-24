import os
import json
import torch
import random
import argparse
import numpy as np
from dotmap import DotMap

from src.utils.utils import load_config
from src.datasets.dataset_annotated_timing import get_dataloader, get_dataset
from src.utils.trainer_timing import trainer
from src.utils.tester_timing import tester
from src.models.timing.model_proposed import System
from src.models.timing.model_baseline import BaselineSystem

EXP_ROOT = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/exp/annotated/data_-500_2000/text_generation/1lstm'

# python scripts/timing/run_annotated_timing2.py configs/timing/annotated_timing_baseline_mla_s1234.json --model baseline --gpuid 0
# python scripts/timing/run_annotated_timing2.py configs/timing/annotated_timing_baseline_tg_mla_s1234.json --model baseline --gpuid 0
# python scripts/timing/run_annotated_timing2.py configs/timing/annotated_timing_proposed_mla_s1234.json --model proposed --gpuid 0


SPEAKERS = ['M1']
FILES = ['M1_all']
TRAIN_FILES = ['M1_train']
VALID_FILES = ['M1_train']
TEST_FILES = ['M1_test']


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def cross_validation(config, model, device, cv_id, out_name):  
    
    config.text_generation.text_generation_lstm_model_path = os.path.join(EXP_ROOT, 'cv{}'.format(str(cv_id)), 'best_val_loss_model.pth')
          
    train_dataset = get_dataset(config, cv_id, split='train', subsets=FILES, speaker_list=SPEAKERS)
    val_dataset = get_dataset(config, cv_id, split='test', subsets=FILES, speaker_list=SPEAKERS)   
    train_loader = get_dataloader(train_dataset, config, shuffle=True)
    val_loader = get_dataloader(val_dataset, config, shuffle=False)  
    loader_dict = {'train': train_loader, 'val': val_loader}
    del train_dataset
    del val_dataset
    
    if model == 'baseline':
        model = BaselineSystem(config, device)
    else:
        model = System(config, device)
    model.to(device)
    
    parameters = model.configure_optimizer_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.optim_params.learning_rate,
        weight_decay=config.optim_params.weight_decay,
    )
    
    os.makedirs(os.path.join(config.exp_dir, out_name), exist_ok=True)
    
    trainer(
        num_epochs=config.num_epochs,
        model=model,
        loader_dict=loader_dict,
        optimizer=optimizer,
        device=device,
        outdir=os.path.join(config.exp_dir, out_name),
        phasename=out_name
    )
    
    tester(
        config,
        device,
        loader_dict['val'],
        model,
        model_dir=os.path.join(config.exp_dir, out_name),
        out_dir=config.exp_dir,
        resume_name=out_name,
        resume=True,
    )


def run(config, model, device, cv_id, out_name):
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
    
    if model == 'baseline':
        model = BaselineSystem(config, device)
    else:
        model = System(config, device)
    model.to(device)
    
    parameters = model.configure_optimizer_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.optim_params.learning_rate,
    )
    
    os.makedirs(os.path.join(config.exp_dir, out_name), exist_ok=True)
    
    trainer(
        num_epochs=config.num_epochs,
        model=model,
        loader_dict=loader_dict,
        optimizer=optimizer,
        device=device,
        outdir=os.path.join(config.exp_dir, out_name),
        phasename=out_name
    )
    
    tester(
        config,
        device,
        loader_dict['train'],
        model,
        model_dir=os.path.join(config.exp_dir, out_name),
        out_dir=config.exp_dir,
        resume_name=out_name,
        resume=True,
    )
    

def set_config_device(args):
    config = load_config(args.config)
    seed_everything(config.seed)
    
    if config.cuda:
        device = torch.device('cuda:{}'.format(args.gpuid))
    else:
        device = torch.device('cpu')

    cross_validation(config, args.model, device, 1, 'cv1')
    cross_validation(config, args.model, device, 2, 'cv2')
    cross_validation(config, args.model, device, 3, 'cv3')
    cross_validation(config, args.model, device, 4, 'cv4')
    cross_validation(config, args.model, device, 5, 'cv5')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs/config.path', help='path to config file')
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'proposed'], help='model type')
    parser.add_argument('--gpuid', type=int, default=-1, help='gpu device id')
    args = parser.parse_args()
    set_config_device(args)
    
