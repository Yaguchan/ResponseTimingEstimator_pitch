import os
import json
import torch
import argparse
import numpy as np
from dotmap import DotMap
from src.datasets.dataset_annotated_vad import get_dataloader, get_dataset
from src.utils.utils import load_config
from src.utils.trainer_vad_and_f0 import trainer, tester
from src.models.vad_and_f0.model_vad_and_f0 import VAD_AND_F0_Predictor


# python scripts/vad_and_f0/run_annotated_vad_and_f0.py configs/vad_and_f0/annotated_vad_and_f0.json --gpuid 0


def run(args):
    config = load_config(args.config)
        
    if config.cuda:
        device = torch.device('cuda:{}'.format(args.gpuid))
    else:
        device = torch.device('cpu')
    
    train_dataset = get_dataset(config, 'train')
    val_dataset = get_dataset(config, 'valid')
    test_dataset = get_dataset(config, 'test')
    
    train_loader = get_dataloader(train_dataset, config, 'train')
    val_loader = get_dataloader(val_dataset, config, 'valid')
    test_loader = get_dataloader(test_dataset, config, 'test')
    
    loader_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    del train_dataset
    del val_dataset
    del test_dataset
    
    model = VAD_AND_F0_Predictor(config, device)
    model.to(device)
    
    parameters = model.configure_optimizer_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.optim_params.learning_rate
    )
    
    os.makedirs(config.exp_dir, exist_ok=True)
    
    trainer(
        num_epochs=config.num_epochs,
        model=model,
        loader_dict=loader_dict,
        optimizer=optimizer,
        device=device,
        outdir=config.exp_dir,
    )
    
    tester(
        model=model,
        loader_dict=loader_dict,
        device=device,
    )

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs/config.path', help='path to config file')
    parser.add_argument('--gpuid', type=int, default=-1, help='gpu device id')
    args = parser.parse_args()
    run(args)