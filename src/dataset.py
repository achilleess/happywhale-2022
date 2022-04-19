import os
import random
import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset
import albumentations
import cv2
import numpy as np
import pandas as pd

from .utils import split_mlp


class HappyWhaleDataset(Dataset):
    def __init__(self, config, annotations, use_aug=False, test=False, dataset_mode=None):
        self.config = config
        self.annotations = annotations
        self.use_aug = use_aug
        self.img_size = config.train_procedure['img_sizes'][-1]

        self.dataset_mode = dataset_mode
        if test:
            self.img_dir_fullbody = config.dataset['test_img_dir_fullbody']
            self.img_dir_backfin = config.dataset['test_img_dir_backfin']
        else:
            self.img_dir_fullbody = config.dataset['train_img_dir_fullbody']
            self.img_dir_backfin = config.dataset['train_img_dir_backfin']

        with open('extra_data/name_to_id.json', 'r') as f:
            self.name_to_id = json.load(f)

        self.aug = albumentations.Compose([
            albumentations.HorizontalFlip(),
            #albumentations.augmentations.geometric.transforms.Affine(
            #    scale=(0.8, 1.2),
            #    rotate=(-7, 7),
            #    shear=(-6, 6),
            #    mode=cv2.BORDER_CONSTANT,
            #    cval=0,
            #    interpolation=cv2.INTER_CUBIC,
            #    p=1
            #),
            albumentations.ColorJitter(
                hue=0.001,
                saturation=0.3,
                contrast=0.2,
                brightness=0.1,
                p=1
            ),
        ])

        self.class_weights = defaultdict(lambda: 1)

        betta = 0.37
        if 'individual_id' in annotations:
            for ind_id, counter in annotations['individual_id'].value_counts().items():
                self.class_weights[ind_id] = 1#(1 - betta) / (1 - betta ** counter)

    def __len__(self):
        return self.annotations.shape[0]
    
    def __getitem__(self, index):
        row = self.annotations.iloc[index]

        img_name = row.image
        if self.dataset_mode == 'backfin':
            img_path = os.path.join(self.img_dir_backfin, img_name)
        elif self.dataset_mode == 'fullbody':
            img_path = os.path.join(self.img_dir_fullbody, img_name)
        else:
            p = random.random()
            if p > 0.3:
                img_path = os.path.join(self.img_dir_backfin, img_name)
            else:
                img_path = os.path.join(self.img_dir_fullbody, img_name)
        
        # Read and transform the image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = cv2.resize(image, (self.img_size, self.img_size))

        if self.use_aug:
            image = self.aug(image=image)['image']

        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0

        ret_dict = {
            'images': image,
            'image_codes': img_name,
        }

        if 'individual_id' in row:
            ret_dict['targets'] = self.name_to_id[img_name][1]
            ret_dict['weights'] = self.class_weights[row.individual_id]
        else:
            ret_dict['targets'] = -1
            ret_dict['weights'] = 1
        return ret_dict
    
    def set_stage(self, stage):
        self.img_size = self.config.train_procedure['img_sizes'][stage]


def get_test_loader(config, test, ind_id_map):
    test_dataset = HappyWhaleDataset(config,
        annotations=test,
        ind_id_map=ind_id_map,
        use_aug=False,
        test=True,
        dataset_mode=config.test_dataset_mode
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    return test_loader


def get_loaders(config, df, train_i, valid_i, train_aug=False, datasets_only=False):
    train_df = df.iloc[train_i, :]
    valid_df = df.iloc[valid_i, :]

    # Datasets & Dataloader
    train_dataset = HappyWhaleDataset(
        config=config,
        annotations=train_df,
        use_aug=train_aug,
        dataset_mode=config.dataset['train_mode']
    )
    valid_dataset = HappyWhaleDataset(
        config=config,
        annotations=valid_df,
        use_aug=False,
        dataset_mode=config.dataset['val_mode']
    )
    
    if datasets_only:
        return train_dataset, valid_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset['batch_size'],
        num_workers=config.dataset['num_workers'],
        shuffle=True,
        drop_last=False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.dataset['batch_size'],
        num_workers=config.dataset['num_workers'],
        shuffle=False,
        drop_last=False
    )
    return train_loader, valid_loader


class MLPDataset(Dataset):
    def __init__(self, features, names, sources):
        self.features = features
        self.names = names
        self.sources = sources
        
        with open('extra_data/name_to_id.json', 'r') as f:
            self.name_to_id = json.load(f)

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        name = self.names[index]

        ret_dict = {}
        for source in self.sources:
            ret_dict[source] = self.features[source][name]

        ret_dict['image_codes'] = name
        ret_dict['targets'] = -1
        if name in self.name_to_id:
            ret_dict['targets'] = int(self.name_to_id[name][1])
        return ret_dict


def load_mlp_features(sources, train=True):
    ret_dict = {}
    file_name = 'train.csv' if train else 'test.csv'
    log_tag = 'training' if train else 'testing'
    for source in sources:
        print(f'Loading {log_tag} {source}...')
        file_path = os.path.join('preload', source, file_name)
        df = pd.read_csv(file_path)
        name_feat_map = {}
        for name, emb_str in zip(df['names'], df['embs']):
            emb = emb_str.replace('\n', ' ').replace('[', '').replace(',', '').replace(']', '').split()
            name_feat_map[name] = np.array(emb).astype(np.float32)
        ret_dict[source] = name_feat_map
    return ret_dict


def build_mlp_test_dataloader(config, fold):
    features = load_mlp_features(config.dataset['sources'], train=False)
    inter_names = set()
    for name in features:
        if not inter_names:
            inter_names = set(features[name].keys())
        else:
            inter_names.intersection(set(features[name].keys()))
    inter_names = sorted(list(inter_names))

    assert len(inter_names) == 27956
    
    test_dataset = MLPDataset(features, inter_names, config.dataset['sources'])

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        config.dataset['batch_size'],
        shuffle=False,
        num_workers=config.dataset['num_workers'], 
        drop_last=False
    )
    return test_loader


def get_valid_targets(train_dataset):
    train_targets = set([int(train_dataset.name_to_id[i][1]) for i in train_dataset.names])
    return train_targets


def build_mlp_dataloaders(config, fold, with_valid_targets=False):
    features = load_mlp_features(config.dataset['sources'])

    inter_names = set()
    for name in features:
        if not inter_names:
            inter_names = set(features[name].keys())
        else:
            inter_names = inter_names.intersection(set(features[name].keys()))
    inter_names = sorted(list(inter_names))

    print(f'Training features have {len(inter_names)} common dataset sampels')

    train_names, val_names = split_mlp(inter_names, fold, config)

    train_dataset = MLPDataset(features, train_names, config.dataset['sources'])
    val_dataset = MLPDataset(features, val_names, config.dataset['sources'])


    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        config.dataset['batch_size'],
        shuffle=True,
        num_workers=config.dataset['num_workers'],
        drop_last=False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        config.dataset['batch_size'],
        shuffle=False,
        num_workers=config.dataset['num_workers'],
        drop_last=False
    )
    if with_valid_targets:
        valid_targets = get_valid_targets(train_dataset)
        return train_loader, val_loader, valid_targets
    else:
        return train_loader, val_loader