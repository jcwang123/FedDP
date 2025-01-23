import math, sys, os

sys.path.insert(0, os.path.dirname(__file__) + '/../../')

import numpy as np
from networks.Calibration.modules import HeadCalibration, PersonalizedChannelSelection
from networks.Calibration.soft_attn import Soft, get_gaussian_kernel
import torch.nn as nn
import torch.nn.functional as F
import torch


class _LCSegmentationModel(nn.Module):
    # general segmentation model
    def __init__(self, trans, backbone, decoder, head):
        super(_LCSegmentationModel, self).__init__()
        self.trans = trans
        self.backbone = backbone
        self.head = head
        self.decoder = decoder
        self.pcs = PersonalizedChannelSelection(f_dim=256, emb_dim=64)

    def forward(self, x, emb=None):
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        features[-1] = self.pcs(features[-1])
        x = self.decoder(features[-4], features[-3], features[-2],
                         features[-1])
        x = self.head(x)
        x = F.interpolate(x,
                          size=input_shape,
                          mode='bilinear',
                          align_corners=False)
        return x