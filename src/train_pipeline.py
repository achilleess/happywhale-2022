from datetime import datetime
import time
import gc
import os

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
import torch.optim as optims

from .dataset import get_loaders
from .model import CNNModel
from .stat import StatKeeper
from .utils import set_seed, to_device, get_lr_callback


def forward_step(model, batch, criterion, stat_keeper, 
                    config, scaler=None, optimizer=None, batch_num=None):
    images, targets, weights = to_device(
        [batch['images'], batch['targets'], batch['weights']], config)

    if not scaler is None:
        with torch.cuda.amp.autocast():
            out, _ = model(images, targets)
    else:
        out, _ = model(images, targets)

    weights = weights / sum(weights) * weights.size(0)
    loss = (criterion(out, targets) * weights).mean()

    stat_keeper.step(torch.argmax(out, dim=1), targets, loss)
    loss = loss / config.train_procedure['accum_grad_steps']

    if not optimizer is None:
        if not scaler is None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        do_update = batch_num % config.train_procedure['accum_grad_steps'] == 0
        if not scaler is None and do_update:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        elif do_update:
            optimizer.step()
            optimizer.zero_grad()
    return


def validate_model(config, model, loader, criterion, scaler=None):
    # === EVAL ===
    stat_keeper = StatKeeper()
    model.eval() 
    for batch_idx, batch in enumerate(tqdm(loader)):
        with torch.no_grad():
            loss = forward_step(model, batch, criterion,
                stat_keeper, scaler=scaler, config=config
            )
    return stat_keeper


def train_fold(config, model, optimizer, train_loader, valid_loader, criterion, fold_n):
    best_score = -1

    model_dir = config.train_procedure['model_save_dir']
    best_model_path = os.path.join(model_dir, f'fold_{fold_n}_best.pth')
    epoch_model_path = os.path.join(model_dir, f'fold_{fold_n}_last.pth')

    if config.train_procedure['resume']:
        model.load_state_dict(torch.load(epoch_model_path))

    scaler = torch.cuda.amp.GradScaler() if config.train_procedure['fp16'] else None
    start_epoch = 0 if not config.train_procedure['resume'] else config.train_procedure['resume_epoch']

    for epoch in range(start_epoch, config.train_procedure['epochs']):
        
        if epoch in config.train_procedure['stage_epochs']:
            stage_id = config.train_procedure['stage_epochs'].index(epoch)
            train_loader.dataset.set_stage(stage_id)
            valid_loader.dataset.set_stage(stage_id)

        get_lr_callback(epoch, config, optimizer)

        print("~"*8, f"Epoch {epoch + 1}", "~"*8)
        print("learning rate: ")
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            if config.train_procedure['use_wandb']:
                wandb.log({'lr': param_group['lr']}, step=epoch)
            
        stat_keeper = StatKeeper()
        
        progressbar = tqdm(train_loader, desc='TRAIN Epoch {}/{}'.format(
            epoch + 1, config.train_procedure['epochs']))

        model.train()
        for batch_num, batch in enumerate(progressbar):
            #if batch_num == 40:
            #    break
            loss = forward_step(model, batch, criterion, stat_keeper,
                 scaler=scaler, optimizer=optimizer, config=config, batch_num=batch_num
            )
            progressbar.set_postfix(loss=stat_keeper.get_running_loss())

        optimizer.zero_grad()

        progressbar.close()
        time.sleep(0.3)

        train_stat = stat_keeper.reset(print_tag='Traning statistics:')
        if config.train_procedure['use_wandb']:
            wandb.log({
                "mean_train_loss": train_stat['avg_loss'],
                "train_avg_recall": train_stat['avg_recall'],
                "train_acc": train_stat['acc']}, step=epoch)
        
        gc.collect()

        # === EVAL ===
        model.eval()
        stat_keeper = validate_model(config, model, valid_loader, criterion, scaler)
        
        val_stat = stat_keeper.reset(print_tag='Valid statistics')
        if config.train_procedure['use_wandb']:
            wandb.log({
                "mean_valid_loss": val_stat['avg_loss'],
                "val_avg_recall": val_stat['avg_recall'],
                "val_acc": val_stat['acc']}, step=epoch)

        if val_stat['acc'] > best_score:        
            print("!! Saving best model !!\n")
            torch.save(model.state_dict(), best_model_path)
            best_score = val_stat['acc']
        torch.save(model.state_dict(), epoch_model_path)
        print()

        gc.collect()
    return

    
def train_pipeline(flags):
    config, train = flags['config'], flags['train']
    skf_splits, fold_to_train = flags['skf_splits'], flags['fold_to_train']

    set_seed()

    for fold_i, (train_i, valid_i) in enumerate(skf_splits):
        if fold_i == fold_to_train - 1:
            break

    run = None
    if config.train_procedure['use_wandb']:
        now = datetime.now()
        run = wandb.init(
            project='happywhale',
            name=now.strftime("%H:%M:%S") + f'_fold_{fold_i}',
            config=config
        )

    print("~" * 25)
    print("~" * 8, f"FOLD {fold_i}", "~" * 8)
    print("~" * 25)
        
    train_loader, valid_loader = get_loaders(
        config=config,
        df=train,
        train_i=train_i,
        valid_i=valid_i,
        train_aug=True
    )

    model = CNNModel(config).to(config.device)
    criterion = nn.CrossEntropyLoss(reduction='none')

    optimizer_cls = getattr(optims, config.optimizer.pop('type'))
    optimizer = optimizer_cls(params=model.parameters(), **config.optimizer)

    train_fold(config, model, optimizer, train_loader, valid_loader, criterion, fold_i)

    gc.collect()


# it was distributed training one time :)
def spawn_train_processes(config, train, skf_splits, fold_to_train):
    flags = {
        'config': config,
        'train': train,
        'skf_splits': skf_splits,
        'fold_to_train': fold_to_train
    }

    train_pipeline(flags=flags)