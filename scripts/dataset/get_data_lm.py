import os
import json
import torch
import random
import argparse
import numpy as np
from dotmap import DotMap
from tqdm import tqdm
from transformers import GPT2Tokenizer
from src.utils.utils import load_config
from src.datasets.dataset_annotated_text_generation import get_dataloader, get_dataset



# create dataset
# python scripts/dataset/get_data_lm.py configs/dataset/config.json --gpuid 0 --cv_id 1

# GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# gpt2 preparation
# pip install git+https://github.com/huggingface/transformers
# git clone https://github.com/huggingface/transformers
# pip install -r ./transformers/examples/pytorch/language-modeling/requirements.txt
# pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0

# gpt2 train
# cv_1
# python ./transformers/examples/pytorch/language-modeling/run_clm.py --model_name_or_path=gpt2 --train_file=data/text_generation/cv_1/train.txt --validation_file=data/text_generation/cv_1/valid.txt --do_train --do_eval --block_size=256 --num_train_epochs=100 --save_steps=1000 --save_total_limit=3 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --output_dir=exp/text_generation/gpt2/epoch100/cv_1 --use_fast_tokenizer=False
# cv_2
# python ./transformers/examples/pytorch/language-modeling/run_clm.py --model_name_or_path=gpt2 --train_file=data/text_generation/cv_2/train.txt --validation_file=data/text_generation/cv_2/valid.txt --do_train --do_eval --block_size=256 --num_train_epochs=100 --save_steps=1000 --save_total_limit=3 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --output_dir=exp/text_generation/gpt2/epoch100/cv_2 --use_fast_tokenizer=False
# cv_3
# python ./transformers/examples/pytorch/language-modeling/run_clm.py --model_name_or_path=gpt2 --train_file=data/text_generation/cv_3/train.txt --validation_file=data/text_generation/cv_3/valid.txt --do_train --do_eval --block_size=256 --num_train_epochs=100 --save_steps=1000 --save_total_limit=3 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --output_dir=exp/text_generation/gpt2/epoch100/cv_3 --use_fast_tokenizer=False
# cv_4
# python ./transformers/examples/pytorch/language-modeling/run_clm.py --model_name_or_path=gpt2 --train_file=data/text_generation/cv_4/train.txt --validation_file=data/text_generation/cv_4/valid.txt --do_train --do_eval --block_size=256 --num_train_epochs=100 --save_steps=1000 --save_total_limit=3 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --output_dir=exp/text_generation/gpt2/epoch100/cv_4 --use_fast_tokenizer=False
# cv_5
# python ./transformers/examples/pytorch/language-modeling/run_clm.py --model_name_or_path=gpt2 --train_file=data/text_generation/cv_5/train.txt --validation_file=data/text_generation/cv_5/valid.txt --do_train --do_eval --block_size=256 --num_train_epochs=100 --save_steps=1000 --save_total_limit=3 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --output_dir=exp/text_generation/gpt2/epoch100/cv_5 --use_fast_tokenizer=False


SPEAKERS = ['M1']
FILES = ['M1_all']
TRAIN_FILES = ['M1_train']
VALID_FILES = ['M1_train']
TEST_FILES = ['M1_test']

OUTDIR = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/data/text_generation'


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def cross_validation_get_text(loader, cv_id, split):
    list_data = []
    
    for batch in tqdm(loader):
        for i in range(len(batch[0])):
            texts = batch[0][0].replace('<eou>', tokenizer.eos_token)
            list_data.append(texts)
    
    path = os.path.join(OUTDIR, f'cv_{cv_id}')
    os.makedirs(path, exist_ok=True)     
    with open(os.path.join(path, f'{split}.txt'), mode='w') as f:
        for data in list_data:
            f.write(data + "\n")


def all_get_text(loader, split):
    list_data = []
    
    for batch in tqdm(loader):
        for i in range(len(batch[0])):
            texts = batch[0][0].replace('<eou>', tokenizer.eos_token)
            list_data.append(texts)
    
    path = os.path.join(OUTDIR, 'all')
    os.makedirs(path, exist_ok=True)     
    with open(os.path.join(path, f'{split}.txt'), mode='w') as f:
        for data in list_data:
            f.write(data + "\n")


def cross_validation(config, device, cv_id, out_name):        
    train_dataset = get_dataset(config, cv_id, split='train', subsets=FILES, speaker_list=SPEAKERS)
    val_dataset = get_dataset(config, cv_id, split='test', subsets=FILES, speaker_list=SPEAKERS)   
    train_loader = get_dataloader(train_dataset, config, shuffle=True)
    val_loader = get_dataloader(val_dataset, config, shuffle=False)
    del train_dataset
    del val_dataset
    cross_validation_get_text(train_loader, cv_id, 'train')
    cross_validation_get_text(val_loader, cv_id, 'valid')


def run(config, device, cv_id, out_name):
    train_dataset = get_dataset(config, cv_id, split='train', subsets=FILES, speaker_list=SPEAKERS)
    val_dataset = get_dataset(config, cv_id, split='valid', subsets=FILES, speaker_list=SPEAKERS)
    test_dataset = get_dataset(config, cv_id, split='test', subsets=FILES, speaker_list=SPEAKERS) 
    train_loader = get_dataloader(train_dataset, config, shuffle=True)
    val_loader = get_dataloader(val_dataset, config, shuffle=False)
    test_loader = get_dataloader(test_dataset, config, shuffle=False)
    del train_dataset
    del val_dataset
    del test_dataset
    all_get_text(train_loader, 'train')
    all_get_text(val_loader, 'valid')
    all_get_text(test_loader, 'test')
    

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
    cross_validation(config, device, int(args.cv_id), 'cv{}'.format(int(args.cv_id)))
    
    # all data
    # run(config, device, int(args.cv_id), 'M1_all')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs/config.path', help='path to config file')
    parser.add_argument('--gpuid', type=int, default=-1, help='gpu device id')
    parser.add_argument('--cv_id', type=int, default=0, help='id for the cross validation settings')
    args = parser.parse_args()
    set_config_device(args)