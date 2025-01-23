import math, sys, os

sys.path.insert(0, os.path.dirname(__file__) + '/../../')

import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from networks.Calibration.soft_attn import Soft


class HeadCalibration(nn.Module):
    def __init__(self, n_classes, n_fea):
        super(HeadCalibration, self).__init__()
        self.n_classes = n_classes
        self.head = nn.Conv2d(n_fea * n_classes * 2, n_classes, 1)
        self.soft = Soft()

    def forward(self, uncertainty, preds, fea):
        fea_shape = (fea.size(2), fea.size(3))
        ori_shpae = (uncertainty.size(2), uncertainty.size(3))
        fea_list = []
        att_maps = []
        # print(fea_shape)
        for c in range(self.n_classes):
            fea2 = self.soft(uncertainty[:, c].unsqueeze(1),
                             size=fea_shape) * fea + fea
            fea_list.append(fea2)
            fea3 = self.soft(preds[:, c].unsqueeze(1),
                             size=fea_shape) * fea + fea
            fea_list.append(fea3)
        fea_list = torch.cat(fea_list, dim=1)
        o = self.head(fea_list)
        o = F.sigmoid(o)
        if not fea_shape[0] == ori_shpae[0]:
            o = F.interpolate(o, ori_shpae)
        return o


class PersonalizedChannelSelection(nn.Module):
    def __init__(self, f_dim, emb_dim):
        super(PersonalizedChannelSelection, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Sequential(nn.Conv2d(emb_dim, f_dim, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(f_dim, f_dim, 1, bias=False))
        self.fc2 = nn.Sequential(
            nn.Conv2d(f_dim * 2, f_dim // 16, 1, bias=False), nn.ReLU(),
            nn.Conv2d(f_dim // 16, f_dim, 1, bias=False))

    def forward_emb(self, emb):
        emb = emb.unsqueeze(-1).unsqueeze(-1)
        emb = self.fc1(emb)
        # print(emb.device)
        return emb

    def forward(self, x, emb):
        b, c, w, h = x.size()
        emb = (torch.repeat_interleave(emb, b, 0)).to(x.device)
        # print(emb.shape)

        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        # site embedding
        emb = self.forward_emb(emb)

        # avg
        avg_out = torch.cat([avg_out, emb], dim=1)
        avg_out = self.fc2(avg_out)

        # max
        max_out = torch.cat([max_out, emb], dim=1)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        hmap = self.sigmoid(out)

        x = x * hmap + x

        return x, hmap
