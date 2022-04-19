import json
import random
import os
from collections import Counter, defaultdict

from sklearn.model_selection import StratifiedKFold, KFold
import torch
import pandas as pd
import numpy as np


def set_seed(seed = 12345):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def get_folds(config, train, special_split=False):
    set_seed()

    splits = []
    if not special_split:
        skf = KFold(n_splits=config.train_procedure['n_splits'])
        skf_splits = skf.split(X=train, y=train['individual_id'])
        splits = [(list(train_idxs), list(val_idxs)) for train_idxs, val_idxs in skf_splits]
    else:
        counter = Counter(train['individual_id'].values)
        common_ids = set(name for name, cnt in counter.items() if cnt == 1)

        common_df = train[train['individual_id'].isin(common_ids)]
        train = train[~train['individual_id'].isin(common_ids)]

        skf = StratifiedKFold(n_splits=config.train_procedure['n_splits'])
        skf_splits = skf.split(X=train, y=train['individual_id'])

        common_idxs = [i for i in range(len(train), len(train) + len(common_df))]
        splits = [(list(train_idxs) + common_idxs, list(val_idxs)) for train_idxs, val_idxs in skf_splits]
        print([ (len(i), len(j)) for i, j in splits])

        train = pd.concat([train, common_df])
    return train, splits


def to_device(inps, config):
    for i in range(len(inps)):
        inps[i] = inps[i].to(config.device)
    return inps


def get_lr_callback(epoch, config, optimizer):
    lr_start = 0.000001

    batch_size = config.dataset['batch_size']
    accum_grad_steps = config.train_procedure['accum_grad_steps']
    lr_max = 0.000005 * batch_size  * accum_grad_steps
    lr_min = 0.000001
    lr_ramp_ep = 4
    lr_sus_ep = 0
    lr_decay = 0.90
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        return lr
    
    for g in optimizer.param_groups:
        g['lr'] = lrfn(epoch)


def split_list(fold, n_splits, list_to_split, train_names, val_names, cls_name_to_imgs):
    split_step = len(list_to_split) // n_splits
    start_idx = fold * split_step
    if fold + 1 == n_splits:
        end_idx = len(list_to_split)
    else:
        end_idx = (fold + 1) * split_step
    
    print(start_idx, end_idx)
    for idx, (class_name, _) in enumerate(list_to_split):
        if idx >= start_idx and idx < end_idx:
            val_names.extend(cls_name_to_imgs[class_name])
        else:
            train_names.extend(cls_name_to_imgs[class_name])
    return


def split_mlp(names, fold, config):
    skf = sklearn.model_selection.KFold(n_splits=config.training_procedure['n_splits'])
    skf_splits = skf.split(X=names)

    for fold_idx, (train_idxs, val_idxs) in enumerate(skf_splits):
        if fold_idx == fold:
            return [names[i] for i in train_idxs], [names[i] for i in val_idxs]