import math
from turtle import forward

import torch
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import timm


class ArcMarginProduct(nn.Module):
    def __init__(self, device, in_features, out_features, s=30.0, 
                 m=0.50, easy_margin=False, ls_eps=0.0):
        '''
        in_features: dimension of the input
        out_features: dimension of the last layer (in our case the classification)
        s: norm of input feature
        m: margin
        ls_eps: label smoothing'''
        super(ArcMarginProduct, self).__init__()

        self.in_features, self.out_features = in_features, out_features
        self.s = torch.Tensor([s]).to(device)
        self.m = torch.Tensor([m]).to(device)
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = torch.Tensor([math.cos(math.pi - m)]).to(device)
        self.mm = torch.Tensor([math.sin(math.pi - m) * m]).to(device)
        self.device = device


    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        one_hot = torch.zeros(cosine.size()).to(self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class CNNModel(nn.Module):
    def __init__(self, config):
        super(CNNModel, self).__init__()
        embedding_size = config.model['embedding_size']
        self.backbone = timm.create_model(config.model['model_name'], pretrained=True)

        self.backbone.classifier = nn.Identity()
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.embedding = nn.Sequential(
            nn.Dropout(p=config.model['dropout_rate']),
            nn.Linear(2304, embedding_size, bias=False)
        )
        
        self.arcface = ArcMarginProduct(
            in_features=embedding_size, 
            out_features=config.model['num_classes'],
            device=config.device,
            s=30.0,
            m=0.3,
            easy_margin=False,
            ls_eps=0.0
        )
        
        
    def forward(self, image, target=None):
        features = self.backbone(image)

        embedding = self.embedding(features)
        if not target is None:
            out = self.arcface(embedding, target)
            return out, embedding
        else:
            return embedding


class HiddenBlock(nn.Module):
    def __init__(self, in_features, out_features, use_act=True):
        super(HiddenBlock, self).__init__()
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.SiLU()
        self.lin = nn.Linear(in_features, out_features)
        self.use_act = use_act
    
    def forward(self, x):
        x = self.lin(x)
        if self.use_act:
            x = self.act(x)
        x = self.bn(x)
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        self.feat_map = nn.ModuleDict()
        for feat_name, (in_feat, out_feat) in config.model['features_mapping'].items():
            self.feat_map[feat_name] = nn.Linear(in_feat, out_feat)
        self.concat_feat_names = config.model['concat_features']

        hidden_dim = config.model['hidden_dim']
        input_dim = config.model['input_dim']
        embedding_size = config.model['embedding_size']
        self.body = torch.nn.Sequential(
            nn.Dropout(p=0.15),
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(1024),
        )
        self.end_dp = nn.Dropout(p=0.45)

        self.arcface = ArcMarginProduct(
            in_features=embedding_size, 
            out_features=config.model['num_classes'],
            device=config.device,
            s=30.0,
            m=0.3,
            easy_margin=False,
            ls_eps=0.0
        )
    
    def forward(self, batch, targets=None):
        for feat_name, layer in self.feat_map.items():
            batch[feat_name] = layer(batch[feat_name])
        x = torch.cat([batch[i] for i in self.concat_feat_names], dim=1)

        x = self.body(x)
        if not targets is None:
            x = self.arcface(x, targets)
        return x

    def forward(self, batch, targets=None):
        for feat_name, layer in self.feat_map.items():
            batch[feat_name] = layer(batch[feat_name])
        x = torch.cat([batch[i] for i in self.concat_feat_names], dim=1)

        emb = self.body(x)
        if not targets is None:
            x = self.end_dp(emb)
            x = self.arcface(x, targets)
            return emb, x
        return emb