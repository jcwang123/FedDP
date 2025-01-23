from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from networks.FPN.pvtv2 import pvt_v2_b2, pvt_v2_b0
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
import segmentation_models_pytorch as smp
import timm

from scripts.trainer_utils import compute_pred_uncertainty_by_features
# from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


def BuildFPN(num_classes, encoder='pvtb2', decoder='fpn'):
    if encoder == 'pvtb0':
        backbone = pvt_v2_b0()
        path = 'weights/pvt_v2_b0.pth'
        chs = [32, 64, 160, 256]
        save_model = torch.load(path)
        model_dict = backbone.state_dict()
        state_dict = {
            k: v
            for k, v in save_model.items() if k in model_dict.keys()
        }
        model_dict.update(state_dict)
        backbone.load_state_dict(model_dict)
    elif encoder == 'pvtb2':
        backbone = pvt_v2_b2()
        path = 'weights/pvt_v2_b2.pth'
        chs = [64, 128, 320, 512]
        save_model = torch.load(path)
        model_dict = backbone.state_dict()
        state_dict = {
            k: v
            for k, v in save_model.items() if k in model_dict.keys()
        }
        model_dict.update(state_dict)
        backbone.load_state_dict(model_dict)
    elif encoder == 'resnet50':
        # backbone = timm.models.resnetv2_50(pretrained=True, features_only=True)
        backbone = smp.encoders.get_encoder('resnet50')
        chs = [256, 512, 1024, 2048]
    elif encoder == 'resnet18':
        backbone = smp.encoders.get_encoder('resnet18')
        chs = [64, 128, 256, 512]
    else:
        raise NotImplementedError

    if 'resnet' in encoder:
        trans = False
    else:
        trans = True
    head = _head(num_classes, in_chs=128)
    decoder = FPNDecoder(chs)
    model = _SimpleSegmentationModel(trans, backbone, decoder, head)
    return model


class _head(nn.Module):
    def __init__(self, num_classes, in_chs):
        super(_head, self).__init__()
        self.p_head = nn.Conv2d(in_chs, num_classes, 1)

    def forward(self, feature):
        o = self.p_head(feature)
        o = F.sigmoid(o)
        return o


class _SimpleSegmentationModel(nn.Module):
    # general segmentation model
    def __init__(self, trans, backbone, decoder, head):
        super(_SimpleSegmentationModel, self).__init__()
        self.trans = trans
        self.backbone = backbone
        self.head = head
        self.decoder = decoder
        self.pcs = None
        self.hc = None
        self.memory_all_nets = None
        self.emb = None

    def forward(self,
                x,
                return_features=False,
                return_att_maps=False,
                return_out=False):
        input_shape = x.shape[-2:]
        if self.trans:
            features, maps = self.backbone(x, rt_info=return_att_maps)
        else:
            features = self.backbone(x)
        # print([f.shape for f in features])
        if self.emb is not None:
            assert self.pcs is not None
            features[-1] = self.pcs(features[-1], self.emb)[0]

        out_features = self.decoder(features[-4], features[-3], features[-2],
                                    features[-1])
        if self.memory_all_nets is not None:
            assert self.hc is not None
            un_map, preds = compute_pred_uncertainty_by_features(
                net_clients=self.memory_all_nets, features=out_features)
            outputs = torch.mean(preds, dim=0)
            outputs = self.hc(un_map, outputs.detach(), out_features.detach())
        else:
            outputs = self.head(out_features)
        outputs = F.interpolate(outputs,
                                size=input_shape,
                                mode='bilinear',
                                align_corners=False)
        if return_features:
            return outputs, features
        elif return_out:
            return outputs, out_features
        elif return_att_maps:
            return outputs, maps
        else:
            return outputs


if __name__ == '__main__':
    import os
    # from torchsummary import summary
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    model = BuildFPN(1, 'resnet18', 'fpn').cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1 = model(input_tensor)

    # summary(model, (3, 352, 352))
