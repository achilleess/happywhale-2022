import os
import sys
import argparse

sys.path.append('/root/happawhale')

import torch
import pandas as pd

from src.utils import get_folds, set_device, get_ind_id_map
from src.inference_utils import inference_on_test
from configs.default_config import Config

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preload', action='store_true')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    config = Config()
    set_device(config)

    args = parse_args()

    if not os.path.isdir(config.saving_model_dir):
        assert 0
    if config.fp16 and config.device_name == 'tpu':
        assert 0
    if not os.path.isdir('preload'):
        os.mkdir('preload')
    if not os.path.isdir('submissions'):
        os.mkdir('submissions')


    # Import the data
    train = pd.read_csv(config.train_annos)
    test = pd.read_csv(config.test_annos)

    ind_id_map = get_ind_id_map(train)

    train, skf_splits = get_folds(config, train, special_split=config.special_split)

    inference_on_test(config, train, test, skf_splits, args, ind_id_map)
