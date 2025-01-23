import numpy as np
import torch
from torch.autograd import Variable

import numpy as np


@torch.no_grad()
def compute_pred_uncertainty_by_features(net_clients, features):
    preds = []
    for net in net_clients:
        pred = net(features)
        b, c, h, w = pred.size()
        preds.append(pred.unsqueeze(0))
    preds = torch.cat(preds, dim=0)

    umap = torch.std(preds, dim=0)
    if umap.max() > 0:
        umap = umap / umap.max()
    umap = umap.view(b, c, h, w)
    return umap, preds


@torch.no_grad()
def compute_pred_uncertainty(net_clients, images):
    preds = []
    for net in net_clients:
        pred = net(images)
        b, c, h, w = pred.size()
        preds.append(pred.unsqueeze(0))
    preds = torch.cat(preds, dim=0)

    umap = torch.std(preds, dim=0)
    # print(umap.max())
    if umap.max() > 0:
        umap = umap / umap.max()
    # umap = umap.view(b, c, h * w)
    # umap = torch.softmax(umap, dim=-1)
    umap = umap.view(b, c, h, w)
    return umap, preds


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def set_global_grad(net, keys, tag):
    for name, param in net.named_parameters():
        if name in keys:
            param.requires_grad = (tag == 1)
        else:
            param.requires_grad = (tag == 0)


def check_equal(net_clients):
    params = dict(net_clients[0].named_parameters())
    for name, param in params.items():
        for client in range(1, len(net_clients)):
            _tmp_param_data = dict(
                net_clients[client].named_parameters())[name].data
            assert torch.sum(_tmp_param_data - params[name].data) == 0


def freeze_params(net, keys):
    params = dict(net.named_parameters())
    for name, param in params.items():
        if name in keys:
            dict(net.named_parameters())[name].requires_grad = False
        else:
            dict(net.named_parameters())[name].requires_grad = True


def attack(net_clients, attack_site, ori_params):
    print('Attack Site Gradients Opposite----')
    params = dict(net_clients[attack_site].named_parameters())

    for name, param in params.items():
        new_param_data = params[name].data * 2 - ori_params[name].data
        params[name].data.copy_(new_param_data)


def update_global_model(net_clients, client_weight):
    print('Calculate the model avg----')
    params = dict(net_clients[0].named_parameters())
    for name, param in params.items():
        for client in range(len(net_clients)):
            single_client_weight = client_weight[client]
            if client == 0:
                tmp_param_data = dict(net_clients[client].named_parameters()
                                      )[name].data * single_client_weight
            else:
                tmp_param_data = tmp_param_data + \
                                 dict(net_clients[client].named_parameters())[
                                     name].data * single_client_weight
            params[name].data.copy_(tmp_param_data)
    print('Update each client model parameters----')

    for client in range(len(net_clients)):
        tmp_params = dict(net_clients[client].named_parameters())
        for name, param in params.items():
            tmp_params[name].data.copy_(param.data)


def update_global_model_with_keys(net_clients, client_weight, private_keys):
    print('Calculate the model avg----')
    params = dict(net_clients[0].named_parameters())
    for name, param in params.items():
        for client in range(len(net_clients)):
            single_client_weight = client_weight[client]
            if client == 0:
                tmp_param_data = dict(net_clients[client].named_parameters()
                                      )[name].data * single_client_weight
            else:
                tmp_param_data = tmp_param_data + \
                                 dict(net_clients[client].named_parameters())[
                                     name].data * single_client_weight
            params[name].data.copy_(tmp_param_data)
    print('Update each client model parameters----')

    for client in range(len(net_clients)):
        tmp_params = dict(net_clients[client].named_parameters())
        for name, param in params.items():
            if name in private_keys:
                print('Ignore param: {}'.format(name))
                continue
            tmp_params[name].data.copy_(param.data)


def update_global_model_for_trans(net_clients, client_weight, part=None):
    # 'kv'
    print('Calculate the model avg----')
    params = dict(net_clients[0].named_parameters())
    for name, param in params.items():
        for client in range(len(net_clients)):
            single_client_weight = client_weight[client]
            if client == 0:
                tmp_param_data = dict(net_clients[client].named_parameters()
                                      )[name].data * single_client_weight
            else:
                tmp_param_data = tmp_param_data + \
                                 dict(net_clients[client].named_parameters())[
                                     name].data * single_client_weight
            params[name].data.copy_(tmp_param_data)
    print('Update each client model parameters----')

    for client in range(len(net_clients)):
        tmp_params = dict(net_clients[client].named_parameters())
        for name, param in params.items():
            if part == 'v':
                if '.kv' in name:
                    print('save half the param: {}'.format(name))
                    l = param.data.size()[0]
                    new = torch.cat(
                        [param.data[:l // 2], tmp_params[name].data[l // 2:]],
                        dim=0)
                    tmp_params[name].data.copy_(new)
                else:
                    tmp_params[name].data.copy_(param.data)
            if part == 'k':
                if '.kv' in name:
                    print('save half the param: {}'.format(name))
                    l = param.data.size()[0]
                    new = torch.cat(
                        [tmp_params[name].data[:l // 2], param.data[l // 2:]],
                        dim=0)
                    tmp_params[name].data.copy_(new)
                else:
                    tmp_params[name].data.copy_(param.data)
            if part == 'q':
                if '.q.' in name:
                    print('Ignore param: {}'.format(name))
                    continue
                else:
                    tmp_params[name].data.copy_(param.data)
            if part == 'qk':
                if '.kv' in name:
                    print('save half the param: {}'.format(name))
                    l = param.data.size()[0]
                    new = torch.cat(
                        [tmp_params[name].data[:l // 2], param.data[l // 2:]],
                        dim=0)
                    tmp_params[name].data.copy_(new)
                elif '.q.' in name:
                    print('Ignore param: {}'.format(name))
                    continue
                else:
                    tmp_params[name].data.copy_(param.data)
            if part == 'q_head':
                if '.q.' in name:
                    print('Ignore param: {}'.format(name))
                    continue
                elif '.p_head' in name:
                    print('Ignore param: {}'.format(name))
                    continue
                else:
                    tmp_params[name].data.copy_(param.data)