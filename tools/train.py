import os
import sys
import argparse

sys.path.append('/root/happawhale')

import torch
import pandas as pd

from src.train_pipeline import spawn_train_processes
from src.utils import get_folds
from configs.default_config import Config

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("fold_to_train", type=int)
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()

    config = Config()
    config.device = torch.device(config.train_procedure['device_name'])

    if not os.path.isdir(config.train_procedure['model_save_dir']):
        assert 0

    # Import the data
    train = pd.read_csv(config.dataset['train_annos'])

    train, skf_splits = get_folds(
        config, train, special_split=config.train_procedure['special_split']
    )

    assert args.fold_to_train > 0 and args.fold_to_train <= config.train_procedure['n_splits']

    spawn_train_processes(config, train, skf_splits, args.fold_to_train)