# *_*coding:utf-8 *_*
# late fusion
import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import utils

def train_model(model, data_loader, params):
    # data loader
    train_loader, val_loader, test_loader = data_loader['train'], data_loader['devel'], data_loader['test']
    # criterion
    if params.loss == 'ccc':
        criterion = utils.CCCLoss()
    elif params.loss == 'mse':
        criterion = utils.MSELoss()
    else:
        raise Exception(f'Not supported loss "{params.loss}".')
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.l2_penalty)
    # lr scheduler
    if params.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=params.lr_patience,
                                                 gamma=params.lr_factor)
    else:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                                            patience=params.lr_patience,
                                                            factor=params.lr_factor,
                                                            min_lr=params.min_lr, verbose=True)
    # train
    best_val_loss = float('inf')
    best_val_ccc, best_val_pcc, best_val_rmse = [], [], []
    best_mean_val_ccc = 0
    best_model_file = ''
    early_stop = 0
    for epoch in range(1, params.epochs + 1):
        print('='*50)
        train_loss = train(model, train_loader, criterion, optimizer, epoch, params)
        val_loss, val_ccc, val_pcc, val_rmse = validate(model, val_loader, criterion, params)

        if params.lr_scheduler == 'step':
            lr_scheduler.step()
        else:
            lr_scheduler.step(1 - np.mean(val_ccc))

        mean_val_ccc, mean_val_pcc, mean_val_rmse = np.mean(val_ccc), np.mean(val_pcc), np.mean(val_rmse)
        print('-'*50)
        print(f'Epoch:{epoch:>3} | [Train] | Loss: {train_loss:>.4f}')
        print(f'Epoch:{epoch:>3} |   [Val] | Loss: {val_loss:>.4f} | '
              f'[CCC]: {mean_val_ccc:>7.4f} {[format(x, "7.4f") for x in val_ccc]} | '
              f'PCC: {mean_val_pcc:>.4f} {[format(x, ".4f") for x in val_pcc]} | '
              f'RMSE: {mean_val_rmse:>.4f} {[format(x, ".4f") for x in val_rmse]}')

        if mean_val_ccc > best_mean_val_ccc:
            best_val_ccc = val_ccc
            best_mean_val_ccc = np.mean(best_val_ccc)

            best_model_file = utils.save_model(model, params)
            print(f'Epoch:{epoch:>3} | Save best model "{best_model_file}"!')
            best_val_loss, best_val_pcc, best_val_rmse = val_loss, val_pcc, val_rmse # Note: loss, pcc and rmse when get best val ccc
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= params.early_stop:
                print(f'Note: target can not be optimized for {params.early_stop} consecutive epochs, early stop the training process!')
                break
    best_mean_val_pcc, best_mean_val_rmse = np.mean(best_val_pcc), np.mean(best_val_rmse)
    print('='*50)
    print(f'Best [Val CCC]:{best_mean_val_ccc:>7.4f} {[format(x, "7.4f") for x in best_val_ccc]}| '
          f'Loss: {best_val_loss:>.4f} | '
          f'PCC: {best_mean_val_pcc:>.4f} {[format(x, ".4f") for x in best_val_pcc]} | '
          f'RMSE: {best_mean_val_rmse:>.4f} {[format(x, ".4f") for x in best_val_rmse]}')
    print('='*50)
    
    # predict: val & test
    if params.save:
        print('Predict val & test videos...')
        best_model = torch.load(best_model_file)
        predict(best_model, val_loader, params)
        predict(best_model, test_loader, params)
    else:
        utils.delete_model(best_model_file)

    return best_val_loss, best_val_ccc, best_val_pcc, best_val_rmse


def train(model, train_loader, criterion, optimizer, epoch, params):
    model.train()
    start_time = time.time()
    report_loss, report_size = 0, 0
    total_loss, total_size = 0, 0
    for batch, batch_data in enumerate(train_loader, 1):
        features, feature_lens, labels, metas = batch_data
        batch_size = features.size(0)
        # move to gpu if use gpu
        if params.gpu != None:
            os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
            device = torch.device(f"cuda: {params.gpu}")
            model.cuda(device)
            features = features.cuda(device)
            feature_lens = feature_lens.cuda(device)
            labels = labels.cuda(device)
        optimizer.zero_grad()
        preds = model(features, feature_lens)
        # cal loss
        loss = 0.0
        for i in range(len(params.loss_weights)):
            branch_loss = criterion(preds[:, :, i], labels[:, :, i], feature_lens)
            loss = loss + params.loss_weights[i] * branch_loss
        loss.backward()
        if params.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.clip)
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_size += batch_size

        report_loss += loss.item() * batch_size
        report_size += batch_size

        if batch % params.log_interval == 0:
            avg_loss = report_loss / report_size
            elapsed_time = time.time() - start_time
            print(f"Epoch:{epoch:>3} | Batch: {batch:>3} | Lr: {optimizer.state_dict()['param_groups'][0]['lr']:>1.5f} | "
                  f"Time used(s): {elapsed_time:>.1f} | Training loss: {avg_loss:>.4f}")
            report_loss, report_size, start_time = 0, 0, time.time()

    train_loss = total_loss / total_size
    return train_loss


def validate(model, val_loader, criterion, params):
    model.eval()
    full_preds, full_labels = [], []
    with torch.no_grad():
        val_loss = 0
        val_size = 0
        for batch, batch_data in enumerate(val_loader, 1):
            features, feature_lens, labels, _ = batch_data
            batch_size = features.size(0)
            # move to gpu if use gpu
            if params.gpu != None:
                os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
                device = torch.device(f"cuda: {params.gpu}")
                model.cuda(device)
                features = features.cuda(device)
                feature_lens = feature_lens.cuda(device)
                labels = labels.cuda(device)
            preds = model(features, feature_lens)
            # cal loss
            loss = 0.0
            for i in range(len(params.loss_weights)):
                branch_loss = criterion(preds[:, :, i], labels[:, :, i], feature_lens)
                loss = loss + params.loss_weights[i] * branch_loss
            val_loss += loss.item() * batch_size
            val_size += batch_size

            full_preds.append(preds.cpu().detach().squeeze(0).numpy())
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())
        val_loss /= val_size
        val_ccc, val_pcc, val_rmse  = utils.eval(full_preds, full_labels)

    return val_loss, val_ccc, val_pcc, val_rmse


def predict(model, data_loader, params):
    model.eval()
    full_preds, full_metas = [], []
    with torch.no_grad():
        for batch, batch_data in enumerate(data_loader, 1):
            features, feature_lens, _, metas = batch_data
            # move to gpu if use gpu
            if params.gpu != None:
                os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
                device = torch.device(f"cuda: {params.gpu}")
                model.cuda(device)
                features = features.cuda(device)
                feature_lens = feature_lens.cuda(device)
            preds = model(features, feature_lens)

            full_preds.append(preds.cpu().detach().squeeze(0).numpy())
            full_metas.append(metas.detach().squeeze(0).numpy())

        partition = data_loader.dataset.partition
        utils.write_fusion_result(full_metas, full_preds, params, partition=partition)