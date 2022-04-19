import os
import sys
import argparse
import json
import random
from collections import defaultdict

sys.path.append('/home/nikita/nikita/happawhale')

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optims
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from src.model import MLP
from src.stat import StatKeeperMLP, StatKeeper
from src.dataset import build_mlp_dataloaders
from configs.mlp_config import Config

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("fold_to_train", type=int)
    args = parser.parse_args()
    return args


if __name__=='__main__':
    config = Config()
    config.device = torch.device(config.training_procedure['device_name'])
    args = parse_args()
    fold_to_train = args.fold_to_train

    train_loader, val_loader, valid_targets = \
        build_mlp_dataloaders(config, fold_to_train, with_valid_targets=True)

    model = MLP(config).cuda().train()
    optimizer_cls = getattr(optims, config.optimizer.pop('type'))
    optimizer = optimizer_cls(params=model.parameters(), **config.optimizer)
    lr_drop_epochs = config.training_procedure['lr_drop_epochs']

    epochs = config.training_procedure['epochs']

    #lrs = []
    #scheduler = optims.lr_scheduler.OneCycleLR(
    #    optimizer,
    #    max_lr=0.002,
    #    total_steps=epochs * len(train_loader),
    #    div_factor=1.25,
    #    final_div_factor=12,
    #    pct_start=0.2
    #)

    #for i in range(epochs):
    #    for j in range(train_loader):
    #        scheduler.step()
    #        for g in optimizer.param_groups:
    #            lrs.append(g['lr'])
    #import matplotlib.pyplot as plt
    #plt.plot(lrs)
    #plt.show()

    criterion = torch.nn.CrossEntropyLoss()

    best_acc = -1
    best_model_path = f'models/fold_{fold_to_train}_mlp_best.pth'
    last_model_path = f'models/fold_{fold_to_train}_mlp_last.pth'
    for i in range(epochs):
        print(f'Running {i + 1} epoch...')

        if i in lr_drop_epochs:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 1.5

        progressbar = tqdm(train_loader)
        stat_keeper = StatKeeper()
        model.train()
        train_embs = []
        for batch in progressbar:
            for source in config.dataset['sources']:
                batch[source] = batch[source].to(config.device)
            batch['targets'] = batch['targets'].to(config.device)

            optimizer.zero_grad()
            embs, preds = model(batch, batch['targets'])
            train_embs.append(embs.detach().cpu().numpy())

            loss = criterion(preds, batch['targets'])
            loss.backward()

            stat_keeper.step(torch.argmax(preds, dim=1), batch['targets'], loss)
            optimizer.step()
            #scheduler.step()
            progressbar.set_postfix(loss=stat_keeper.get_running_loss())
        stat_keeper.reset('Training stat')
        
        train_embs = np.concatenate(train_embs, axis=0)
        print('Fitting NearestNeighbors')
        neigh = NearestNeighbors(n_neighbors=1, metric='cosine')
        neigh.fit(train_embs)

        stat_keeper = StatKeeperMLP(neigh)
        model.eval()
        for batch in val_loader:
            for source in config.dataset['sources']:
                batch[source] = batch[source].to(config.device)
            batch['targets'] = batch['targets'].to(config.device)
            
            with torch.no_grad():
                embs, preds = model(batch, batch['targets'])
            loss = criterion(preds, batch['targets'])
            stat_keeper.step(embs, preds, batch['targets'], loss, valid_targets)

        acc = stat_keeper.reset('Validational stat')['acc']
        if acc > best_acc:
            best_acc = acc
            print('Saving best model to - ', best_model_path)
            torch.save(model.state_dict(), best_model_path)
    print('Saving last model to - ', last_model_path)
    torch.save(model.state_dict(), last_model_path)
