from collections import defaultdict
import math
from turtle import distance

import torch
import numpy as np


class StatKeeper():
    def __init__(self):
        self.running_loss = None
        self.correct_preds = 0
        self.counter = 0
        self.losses = []
        self.recall = defaultdict(int)
        self.recall_counter = defaultdict(int)
    
    def step(self, preds, targets, loss):
        preds, targets = preds.flatten(), targets.flatten()
        self.correct_preds += float((preds == targets).sum())
        self.counter += len(preds)

        for p, t in zip(preds, targets):
            p, t = int(p), int(t)
            if p == t:
                self.recall[t] += 1
            self.recall_counter[t] += 1
        
        loss = float(loss)
        if not math.isnan(loss):
            self.losses.append(loss)

            if self.running_loss is None:
                self.running_loss = loss
            else:
                self.running_loss = 0.9 * self.running_loss + 0.1 * loss
    
    def get_running_loss(self):
        return self.running_loss
    
    def reset(self, print_tag=''):
        ret_dict = {}
        ret_dict['acc'] = self.correct_preds / self.counter
        ret_dict['running_loss'] = self.running_loss
        ret_dict['avg_loss'] = sum(self.losses) / len(self.losses)

        avg_recall = 0
        class_counter = 0
        for target_idx, counter in self.recall_counter.items():
            if target_idx < 2000:
                continue
            class_counter += 1
            avg_recall += self.recall[target_idx] / counter
        ret_dict['avg_recall'] = avg_recall / class_counter

        print('\n' + print_tag)
        print('Avg recall', ret_dict['avg_recall'])
        print('Acc: ', ret_dict['acc'])
        print('Average loss: ', ret_dict['avg_loss'])
        return ret_dict


class StatKeeperMLP():
    def __init__(self, neigh):
        self.running_loss = None
        self.correct_preds = 0
        self.counter = 0
        self.losses = []
        self.recall = defaultdict(int)
        self.recall_counter = defaultdict(int)
        self.neigh = neigh

        self.prebs_neg_sum = 0
        self.preds_neg_counter = 0

        self.ss = 0
    
    def step(self, embs, preds, targets, loss, valid_targets):
        embs = embs.detach().cpu().numpy()

        preds = torch.argmax(preds, dim=1).flatten().detach().cpu().numpy()
        targets = targets.flatten().detach().cpu().numpy()

        mask = np.isin(targets, list(valid_targets))
        preds_pos, targets_pos = preds[mask], targets[mask]
        self.correct_preds += float((preds_pos == targets_pos).sum())
        self.counter += len(preds_pos)

        embs_neg = embs[~mask]
        if embs_neg.shape[0]:
            distances, _ = self.neigh.kneighbors(embs_neg, 1, return_distance=True)
            distances = distances.reshape(-1)
    
            self.prebs_neg_sum += sum(distances > 0.55)
            self.preds_neg_counter += len(distances)

        for p, t in zip(preds_pos, targets_pos):
            p, t = int(p), int(t)
            if p == t:
                self.recall[t] += 1
            self.recall_counter[t] += 1
        
        loss = float(loss)
        if not math.isnan(loss):
            self.losses.append(loss)

            if self.running_loss is None:
                self.running_loss = loss
            else:
                self.running_loss = 0.9 * self.running_loss + 0.1 * loss
    
    def get_running_loss(self):
        return self.running_loss
    
    def reset(self, print_tag=''):
        ret_dict = {}
        ret_dict['acc'] = self.correct_preds / self.counter
        ret_dict['running_loss'] = self.running_loss
        ret_dict['avg_loss'] = sum(self.losses) / len(self.losses)

        if self.preds_neg_counter:
            ret_dict['neg_preds_acc'] = self.prebs_neg_sum / self.preds_neg_counter
        else:
            ret_dict['neg_preds_acc'] = None

        avg_recall = 0
        class_counter = 0
        for target_idx, counter in self.recall_counter.items():
            if target_idx < 2000:
                continue
            class_counter += 1
            avg_recall += self.recall[target_idx] / counter
        ret_dict['avg_recall'] = avg_recall / class_counter

        print('\n' + print_tag)
        print('Avg recall', ret_dict['avg_recall'])
        print('Acc: ', ret_dict['acc'])
        print('Acc neg preds', ret_dict['neg_preds_acc'])
        print('Average loss: ', ret_dict['avg_loss'])
        return ret_dict