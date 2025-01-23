from copy import deepcopy
import os
import sys
from numpy.lib.npyio import load

sys.path.insert(0, '/raid0/wjc/PFL-Seg-Trans/')
import argparse
import numpy as np

from scripts.tester_utils import eval_container
from utils.summary import create_logger, DisablePrint, create_summary
from utils.util import load_model

from scripts.train_fed import build_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fl',
                        type=str,
                        default='fedlc_ft_0',
                        help='exp_name')
    parser.add_argument('--net',
                        type=str,
                        default='pvtb0_fpn',
                        help='model_name')
    parser.add_argument('--ver', type=str, default='0', help='version')
    parser.add_argument('--dataset',
                        type=str,
                        default='polyp',
                        help='dataset name')

    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    args = parser.parse_args()

    args.exp = '{}_{}_{}'.format(args.fl, args.net, args.ver)
    args.txt_path = 'logs/{}/{}/txt/'.format(args.dataset, args.exp)
    args.log_path = 'logs/{}/{}/log/'.format(args.dataset, args.exp)
    args.model_path = 'logs/{}/{}/model/'.format(args.dataset, args.exp)
    args.npy_path = 'logs/{}/{}/npy/'.format(args.dataset, args.exp)
    os.makedirs(args.npy_path, exist_ok=True)
    args.logger = create_logger(0, save_dir=args.txt_path)
    print = args.logger.info

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.dataset == 'fundus':
        from dataloaders.fundus_dataloader import Dataset, RandomNoise
        args.client_num = 4
        args.num_classes = 2
        args.c_in = 3
    elif args.dataset == 'pmr':
        from dataloaders.pmr_dataloader import Dataset, RandomNoise
        args.client_num = 6
        args.num_classes = 1
        args.c_in = 1
    elif args.dataset == 'polyp':
        from dataloaders.polyp_dataloader import Dataset, RandomNoise
        args.client_num = 4
        args.num_classes = 1
        args.c_in = 3
    else:
        raise NotImplementedError
    assert args.num_classes > 0 and args.client_num > 1
    args.ds = Dataset
    print(args)
    return args


if __name__ == "__main__":
    # define dataset, model, optimizer for each client
    args = get_args()
    print = args.logger.info
    net = build_model(args)
    dataloader_clients = []
    net_clients = []
    optimizer_clients = []
    for client_idx in range(args.client_num):
        _net = deepcopy(net).cuda()
        _net = load_model(
            _net, args.model_path + '/Site{}_best.pth'.format(client_idx + 1))
        net_clients.append(_net)

    if 'fedlc' in args.fl:
        cur_all_nets = []
        for net in net_clients:
            cur_all_nets.append(net.head)
        for net in net_clients:
            net.memory_all_nets = cur_all_nets

    print('[INFO] Initialized success...')
    _iou = ''
    _dc = ''
    _hd = ''
    _assd = ''
    overall = np.zeros((4, ))
    ious = []

    gen_site = 4
    eval_res = []
    for site_index in range(args.client_num):
        this_net = net_clients[site_index]
        this_net.eval()
        dice_list = []
        dice, haus, iou, assd = eval_container(gen_site,
                                               this_net,
                                               args,
                                               info=True)
        eval_res.append(np.concatenate([dice, haus, iou, assd], axis=0))

        _iou += '&{:.2f}'.format(np.mean(iou) * 100)
        _dc += '&{:.2f}'.format(np.mean(dice) * 100)
        _hd += '&{:.2f}'.format(np.mean(haus))
        _assd += '&{:.2f}'.format(np.mean(assd))
        overall += np.array(
            [np.mean(iou),
             np.mean(dice),
             np.mean(haus),
             np.mean(assd)])
        ious.append(np.mean(iou) * 100)

    eval_res = np.array(eval_res)
    np.save(args.npy_path + f'/site_{gen_site}.npy', eval_res)

    overall /= args.client_num
    context = _iou + '&{:.2f}'.format(
        overall[0] * 100) + _dc + '&{:.2f}'.format(
            overall[1] * 100) + _hd + '&{:.2f}'.format(
                overall[2]) + _assd + '&{:.2f}'.format(overall[3])
    print(_iou + '&{:.2f}'.format(overall[0] * 100))
    print(np.std(ious))
    print(_assd + '&{:.2f}'.format(overall[3]))
    print(context)