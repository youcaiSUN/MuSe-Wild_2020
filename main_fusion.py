# *_*coding:utf-8 *_*
# late fusion
import os
import sys
import glob
from datetime import datetime
from dateutil import tz
import numpy as np
import argparse

import torch
from torch.utils.data.dataset import Dataset

import utils
from dataset import MyDataset
from model import FusionModel
from train_fusion import train_model

# parse parameters
def parse_params():
    parser = argparse.ArgumentParser(description='Late fusion')

    # Data
    parser.add_argument('--base_dir', type=str, required=True,
                        help='base dir for fusion')
    parser.add_argument('--emo_dim_set', nargs='+', default=['arousal'],
                        help='emotion dimensions set ("arousal", "valence")')
    parser.add_argument('--segment_type', type=str, default=None, choices=[None, 'normal', 'id'],
                        help='how to segment video samples?')
    parser.add_argument('--normalize', action='store_true',
                        help='normalize features?')
    parser.add_argument('--win_len', type=int, default=200,
                        help='window length to segment features (default: 200 frames)')
    parser.add_argument('--hop_len', type=int, default=100,
                        help='hop length to segment features (default: 100 frames)')
    # Model
    # Architecture
    parser.add_argument('--d_model', type=int, default=16,
                        help='number of hidden states in the main module (default: 128)')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='number of layers for rnn (default: 1)')
    parser.add_argument('--rnn', type=str, default='lstm', choices=['lstm', 'gru'],
                        help='type of rnn (default: lstm)')
    parser.add_argument('--rnn_bi', action='store_true',
                        help='whether rnn is bidirectional or not (default: False)')
    parser.add_argument('--loss', type=str, default='ccc', choices=['ccc', 'mse'],
                        help='loss function (default: ccc)')
    parser.add_argument('--loss_weights', nargs='+', type=float,
                        help='loss weights for total loss calculation')
    # Dropouts
    parser.add_argument('--dr', type=float, default=0.0,
                        help='dropout rate for main module')
    parser.add_argument('--out_dr', type=float, default=0.0,
                        help='dropout rate for output layer')
    parser.add_argument('--l2_penalty', type=float, default=0.0,
                        help='l2 penalty for network parameter')
    # Tuning
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size (default: 64)')
    parser.add_argument('--early_stop', type=int, default=15,
                        help='number of epochs to early stop training (default: 15)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate (default: 1e-3)')
    parser.add_argument('--lr_scheduler', type=str, default='plateau', choices=['step', 'plateau'],
                        help='scheduler of learning rate (default: plateau)')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='when to decay learning rate (default: 4)')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='learning rate decaying factor (default: 0.5)')
    parser.add_argument('--min_lr', type=float, default=0.0,
                        help='min learning rate when using "ReduceLROnPlateau" (default: 0.5)')
    parser.add_argument('--clip', type=float, default=0,
                        help='gradient clip value (default: 3)')
    # Logistics
    parser.add_argument('--log_interval', type=int, default=1,
                        help='frequency of result logging')
    parser.add_argument('--seed', type=int, default=314,
                        help='random seed')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='number of random seeds')
    parser.add_argument('--gpu', type=str,
                        help='gpu id')
    parser.add_argument('--log', action='store_true',
                        help='whether log file. (default: False)')
    parser.add_argument('--save', action='store_true',
                        help='whether save model and make prediction etc. (default: False)')

    # parse
    args = parser.parse_args()
    return args


def main(params):
    # register logger
    log_file_name = '{}_[FUSION]_[{}]_[{}_{}_{}_{}]_[{}_{}_{}_{}].txt'.format(
        datetime.now(tz=tz.gettz('Asia/Shanghai')).strftime("%Y-%m-%d-%H-%M"),
        '_'.join(params.emo_dim_set),
        params.rnn, params.d_model, params.n_layers, params.rnn_bi,
        params.lr, params.batch_size, params.dr, params.out_dr
    )
    params.log_file_name = log_file_name
    params.log_dir = os.path.join(params.base_dir, 'log')
    if not os.path.exists(params.log_dir):
        os.mkdir(params.log_dir)
    params.log_file = os.path.join(params.log_dir, params.log_file_name)
    if params.log:
        sys.stdout = utils.Logger(params.log_file)
    print(' '.join(sys.argv))
    print(f'Parameters: {params}')

    # check params
    if params.loss_weights is None:
        if len(params.emo_dim_set) == 2:
            params.loss_weights = [0.5, 0.5]  # default: 0.5 * arousal + 0.5 * valence
        else:
            params.loss_weights = [1]
    assert len(params.emo_dim_set) == len(params.loss_weights)

    # load data
    pred_dirs = glob.glob(os.path.join(params.base_dir, 'source/*'))
    data = utils.load_fusion_data(pred_dirs, params.emo_dim_set, params.segment_type)
    print('Constructing dataset and data loader ...')
    data_loader = {}
    for partition in data.keys():
        set_ = MyDataset(data, partition)
        print(f'Samples in "{partition}" set: {len(set_)}')
        batch_size = params.batch_size if partition == 'train' else 1
        shuffle = True if partition == 'train' else False
        data_loader[partition] = torch.utils.data.DataLoader(set_, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    # additional params
    params.d_in = data_loader['train'].dataset.get_feature_dim()
    print(f'Input feature dim: {params.d_in}.')
    params.d_out = len(params.emo_dim_set)

    # seed setting
    seeds = range(params.seed, params.seed + params.n_seeds)
    val_losses, val_cccs, val_pccs, val_rmses = [], [], [], []
    for seed in seeds:
        params.current_seed = seed
        torch.manual_seed(seed)
        if params.gpu is not None and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        print('*' * 100)
        print(f'Using seed "{seed}"')
        print('*' * 100)

        # construct model
        model = FusionModel(params)
        print(model)

        # train model
        print('Training model...')
        val_loss, val_ccc, val_pcc, val_rmse = \
            train_model(model, data_loader, params)

        val_losses.append(val_loss)
        val_cccs.append(val_ccc)
        val_pccs.append(val_pcc)
        val_rmses.append(val_rmse)

        print('*' * 100)
        print(f'Seed "{params.current_seed}" over!')
        print('*' * 100)

    mean_val_cccs = [np.mean(val_ccc) for val_ccc in val_cccs]
    best_idx = mean_val_cccs.index(max(mean_val_cccs))
    best_val_ccc, best_mean_val_ccc = val_cccs[best_idx], mean_val_cccs[best_idx]
    best_val_pcc, best_mean_val_pcc = val_pccs[best_idx], np.mean(val_pccs[best_idx])
    best_val_rmse, best_mean_val_rmse = val_rmses[best_idx], np.mean(val_rmses[best_idx])
    best_val_loss = val_losses[best_idx]
    print(f'Best [Val CCC] when seed "{seeds[best_idx]}":{best_mean_val_ccc:7.4f} {[format(x, "7.4f") for x in best_val_ccc]}')
    print(f'Val PCC when get best CCC: {best_mean_val_pcc:>.4f} {[format(x, ".4f") for x in best_val_pcc]}')
    print(f'Val RMSE when get best CCC: {best_mean_val_rmse:>.4f} {[format(x, ".4f") for x in best_val_rmse]}')
    print(f'Val Loss when get best CCC: {best_val_loss:>.4f}')


if __name__ == '__main__':
    # parse parameters
    params = parse_params()
    main(params)
