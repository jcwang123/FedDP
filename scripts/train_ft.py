import os, sys

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from copy import deepcopy


def main():
    # ------------------  create args ------------------ #
    from config.base import get_args
    args = get_args()
    print = args.logger.info
    print(args)

    from utils.losses import dice_loss, weighted_dice_loss, weighted_bce_loss
    from utils.summary import create_logger, DisablePrint, create_summary

    from scripts.train_fed import build_model, initial_trainer
    from scripts.tester_utils import eval_container
    from scripts.trainer_utils import clip_gradient, compute_pred_uncertainty, compute_pred_uncertainty_by_features
    from tensorboardX import SummaryWriter
    import torch

    # ------------------  initialize the trainer ------------------ #
    dataloader_clients, opt_clients, net_clients = initial_trainer(args)

    # ------------------  initialize the scores ------------------ #
    best_score = np.zeros((len(dataloader_clients), ))
    for site_index in range(args.client_num):
        this_net = net_clients[site_index]
        this_net.eval()
        print("[Test] epoch {} testing Site {}".format(-1, site_index + 1))

        score_values = eval_container(site_index, this_net, args)
        print("Site {} Score {}".format(site_index, np.mean(score_values[0])))
        best_score[site_index] = np.mean(score_values[0])
        save_mode_path = os.path.join(args.model_path,
                                      'Site{}_best.pth'.format(site_index + 1))
        torch.save(net_clients[site_index].state_dict(), save_mode_path)

    # ------------------  start tuning the parameters ------------------ #
    writer = SummaryWriter(args.log_path)
    memory_net_clients = deepcopy(net_clients)
    for epoch_num in range(args.max_epoch):
        if 'bd' in args.fl:
            cur_all_nets = memory_net_clients
        elif 'fedlc' in args.fl:
            cur_all_nets = []
            for net in net_clients:
                cur_all_nets.append(net.head)
        else:
            pass
        for client_idx in range(args.client_num):
            dataloader_current = dataloader_clients[client_idx]
            net_current = net_clients[client_idx]
            if 'fedlc' in args.fl:
                net_current.memory_all_nets = cur_all_nets
            else:
                pass
            net_current.train()
            optimizer_current = opt_clients[client_idx]

            for i_batch, sampled_batch in enumerate(dataloader_current):
                # obtain training data
                volume_batch, label_batch = sampled_batch['image'].cuda(
                ), sampled_batch['label'].cuda()

                # obtain updated parameter at inner loop
                outputs, out_features = net_current(volume_batch,
                                                    return_out=True)

                # obtain the uncertainty
                if 'bd' in args.fl:
                    un_map, preds = compute_pred_uncertainty(
                        net_clients=cur_all_nets, images=volume_batch)

                    plt.imsave(args.model_path + '/un_map.jpg',
                               un_map[0, 0].cpu().numpy(),
                               cmap='hot')
                    seg_loss = dice_loss(outputs, label_batch)
                    bd_loss = weighted_bce_loss(outputs, label_batch, un_map)
                    total_loss = seg_loss + bd_loss * args.lamb
                elif 'fedlc' in args.fl:
                    total_loss = dice_loss(outputs, label_batch)
                else:
                    total_loss = dice_loss(outputs, label_batch)

                optimizer_current.zero_grad()
                total_loss.backward()
                optimizer_current.step()

                iter_num = len(dataloader_current) * epoch_num + i_batch
                if iter_num % 10 == 0:
                    writer.add_scalar('loss/site{}'.format(client_idx + 1),
                                      total_loss, iter_num)
                    if 'bd' in args.fl:
                        writer.add_scalar(
                            'seg_loss/site{}'.format(client_idx + 1), seg_loss,
                            iter_num)
                        writer.add_scalar(
                            'bd_loss/site{}'.format(client_idx + 1), bd_loss,
                            iter_num)
                    print(
                        'Epoch: [%d] client [%d] iteration [%d / %d] : total loss : %f'
                        % (epoch_num, client_idx, iter_num,
                           len(dataloader_current), total_loss.item()))

        for site_index in range(args.client_num):
            this_net = net_clients[site_index]
            this_net.eval()
            print("[Test] epoch {} testing Site {}".format(
                epoch_num, site_index + 1))

            score_values = eval_container(site_index, this_net, args)
            writer.add_scalar('Score/site{}'.format(site_index + 1),
                              np.mean(score_values[0]), epoch_num)
            score = np.mean(score_values[0])
            if score > best_score[site_index]:
                best_score[site_index] = score
                save_mode_path = os.path.join(
                    args.model_path, 'Site{}_best.pth'.format(site_index + 1))
                torch.save(net_clients[site_index].state_dict(),
                           save_mode_path)
            print('[INFO] IoU score {:.4f} Best score {:.4f}'.format(
                np.mean(score_values[0]), best_score[site_index]))

    # ------------------  Final Test ------------------ #
    for site_index in range(args.client_num):
        this_net = net_clients[site_index]
        this_net.eval()
        this_net.load_state_dict(
            torch.load(args.model_path +
                       '/Site{}_best.pth'.format(site_index + 1)))
        dice, haus, iou, assd = eval_container(site_index,
                                               this_net,
                                               args,
                                               info=True)
        np.save(args.npy_path + f'/site_{site_index}.npy',
                np.concatenate([dice, haus, iou, assd], axis=0))


if __name__ == "__main__":
    main()