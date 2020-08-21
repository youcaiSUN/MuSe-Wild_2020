# *_*coding:utf-8 *_*
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
import torch.nn as nn
import config


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()


def load_data(feature_set, emo_dim_set,
              normalize=True,
              norm_opts=None,
              segment_type='normal',
              win_len=100,hop_len=100,
              feature_path=config.PATH_TO_ALIGNED_FEATURES,
              label_path=config.PATH_TO_LABELS,
              save=False,
              refresh=False,
              add_seg_id=False):
    data_file_name = '_'.join(feature_set + emo_dim_set) + f'_{normalize}_{add_seg_id}_{segment_type}_{win_len}_{hop_len}.pkl'
    data_file = os.path.join(config.DATA_FOLDER, data_file_name)
    if os.path.exists(data_file) and not refresh:
        print(f'Find cached data "{os.path.basename(data_file)}".')
        data = pickle.load(open(data_file, 'rb'))
        return data

    print('Constructing data from scratch ...')
    data = {'train': {'feature': [], 'label': [], 'meta': []},
            'devel': {'feature': [], 'label': [], 'meta': []},
            'test' : {'feature': [], 'label': [], 'meta': []}}
    vid2partition, partition2vid = get_data_partition(config.PARTITION_FILE)
    feature_dims = [0] * len(feature_set)
    if add_seg_id == True:
        feature_idx = 1
        print(f'Note: add segment id in the feature.')
    else:
        feature_idx = 2
    for partition, vids in partition2vid.items():
        for vid in vids:
            # concat
            sample_concat_data = [] # feature1, feature2, ..., emo dim1, emo dim2. (ex, 'au', 'vggface', 'arousal', 'valence')
            ## feature
            for i, feature in enumerate(feature_set):
                feature_file = os.path.join(feature_path, feature, vid + '.csv')
                assert os.path.exists(feature_file), f'Error: no available "{feature}" feature file for video "{vid}".'
                df = pd.read_csv(feature_file)
                feature_dims[i] = df.shape[1] - 2
                if i == 0:
                    feature_data = df # keep timestamp and segment id in 1st feature val
                else:
                    feature_data = df.iloc[:, 2:] # feature val starts from third column
                sample_concat_data.append(feature_data)
            ## label
            for emo_dim in emo_dim_set:
                label_file = os.path.join(label_path, emo_dim, vid + '.csv')
                assert os.path.exists(label_file), f'Error: no available "{emo_dim}" label file for video "{vid}".'
                df = pd.read_csv(label_file)
                label = df['value'].values
                label_data = pd.DataFrame(data=label, columns=[emo_dim])
                sample_concat_data.append(label_data)
            # concat
            sample_concat_data = pd.concat(sample_concat_data, axis=1)
            # segment train samples, NOTE: do not segment devel and test samples!
            if partition == 'train':
                samples = segment_sample(sample_concat_data, segment_type, win_len, hop_len) # segmented samples: list
            else:
                samples = [sample_concat_data]
            # store
            for i,segment in enumerate(samples):
                meta = np.column_stack((np.array([int(vid)]*len(segment)), segment.iloc[:,:2].values)) # video id, time stamp, segment id
                data[partition]['meta'].append(meta)
                data[partition]['feature'].append(segment.iloc[:,feature_idx:-len(emo_dim_set)].values) # feature val starts from the "feature_idx"th column
                data[partition]['label'].append(segment.iloc[:,-len(emo_dim_set):].values)

    if normalize: # mainly for audio features
        idx_list = []
        if add_seg_id: # norm seg id
            feature_dims = [1] + feature_dims
            feature_set = ['seg_id'] + feature_set
        assert norm_opts is not None and len(norm_opts) == len(feature_set)
        norm_opts = [True if norm_opt == 'y' else False for norm_opt in norm_opts]
        print('Feature dims: ', feature_dims)
        feature_dims = np.cumsum(feature_dims).tolist()
        feature_dims = [0] + feature_dims
        feature_idxs = zip(feature_dims[0:-1], feature_dims[1:])
        norm_feature_set = []
        for i, (s_idx, e_idx) in enumerate(feature_idxs):
            norm_opt, feature = norm_opts[i], feature_set[i]
            if norm_opt == True:
                norm_feature_set.append(feature)
                idx_list.append([s_idx, e_idx])
        print('Normalize features: ', norm_feature_set)
        print('Indices of normalized features: ', idx_list)
        data = normalize_data(data, idx_list)
    # save data
    if save:
        print('Dumping data...')
        pickle.dump(data, open(data_file, 'wb'))
    return data


def load_fusion_data(pred_dirs, emo_dim_set,
                     segment_type=None,
                     win_len=200, hop_len=100,
                     normalize=False,
                     label_path=config.PATH_TO_LABELS):
    print('Constructing fusion data from scratch ...')
    data = {'train': {'feature': [], 'label': [], 'meta': []},
            'devel': {'feature': [], 'label': [], 'meta': []},
            'test' : {'feature': [], 'label': [], 'meta': []}}
    vid2partition, partition2vid = get_data_partition(config.PARTITION_FILE)
    for partition, vids in partition2vid.items():
        for vid in vids:
            # concat
            sample_concat_data = [] # pred_1 emo_dim_1, pred_1 emo_dim_2, ..., label emo_dim_1, label emo_dim_2.
            ## preds
            first = True
            for pred_dir in pred_dirs:
                for emo_dim in emo_dim_set: # concat emo dim
                    pred_file = os.path.join(pred_dir, f'csv/{emo_dim}/{vid}.csv')
                    assert os.path.exists(pred_file), f'Error: no available prediction file for video "{vid}" in "{pred_dir}".'
                    try:
                        df = pd.read_csv(pred_file)
                    except Exception as e:
                        print(e)
                        print(pred_file)
                        exit()
                    if first == True:
                        cols = list(df) # timestamp, value, segment_id
                        cols[1], cols[2] = cols[2], cols[1] # exchange value and segment_id
                        feature_data = df.loc[:,cols] # keep timestamp and segment id in 1st feature val
                        first = False
                    else:
                        feature_data = df.iloc[:, 1] # prediction value in second column
                    sample_concat_data.append(feature_data)
            ## label
            for emo_dim in emo_dim_set:
                label_file = os.path.join(label_path, emo_dim, vid + '.csv')
                assert os.path.exists(label_file), f'Error: no available "{emo_dim}" label file for video "{vid}".'
                df = pd.read_csv(label_file)
                label_data = df.iloc[:, [1]].rename(columns={'value': emo_dim})  # label value is in second column
                sample_concat_data.append(label_data)
            # concat
            sample_concat_data = pd.concat(sample_concat_data, axis=1)
            sample_concat_data = sample_concat_data.reset_index(drop=True)
            # segment train samples, NOTE: do not segment devel and test samples!
            if partition == 'train' and segment_type is not None:
                samples = segment_sample(sample_concat_data, segment_type, win_len, hop_len) # segmented samples: list
            else:
                samples = [sample_concat_data]
            # store
            for i,segment in enumerate(samples):
                meta = np.column_stack((np.array([int(vid)]*len(segment)), segment.iloc[:,:2].values)) # video id, time stamp, segment id
                data[partition]['meta'].append(meta)
                data[partition]['feature'].append(segment.iloc[:,2:-len(emo_dim_set)].values)
                data[partition]['label'].append(segment.iloc[:,-len(emo_dim_set):].values)

    if normalize:
        input_dim = data['train']['feature'][0].shape[1]
        assert input_dim == len(emo_dim_set) * len(pred_dirs)
        idx_list = [0, input_dim]
        data = normalize_data(data, idx_list)

    return data


def normalize_data(data, idx_list):
    if len(idx_list) == 0: # modified
        return data
    train_concat_data = np.row_stack(data['train']['feature'])
    train_mean = np.mean(train_concat_data, axis=0)
    train_std = np.std(train_concat_data, axis=0)
    for partition in data.keys():
        for i in range(len(data[partition]['feature'])):
            for s_idx, e_idx in idx_list:
                data[partition]['feature'][i][:, s_idx:e_idx] = \
                    (data[partition]['feature'][i][:, s_idx:e_idx] - train_mean[s_idx:e_idx]) / (train_std[s_idx:e_idx] + config.EPSILON)

    return data


def segment_sample(sample, segment_type, win_len, hop_len=None, is_training=False):
    segmented_sample = []
    if hop_len is None:
        hop_len = win_len
    else:
        assert hop_len <= win_len
    if segment_type == 'id':
        segment_ids = sorted(set(sample['segment_id'].values))
        for id in segment_ids:
            segment = sample[sample['segment_id']==id]
            for s_idx in range(0, len(segment), hop_len):
                e_idx = min(s_idx+win_len, len(segment))
                sub_segment = segment.iloc[s_idx:e_idx]
                segmented_sample.append(sub_segment)
                if e_idx == len(segment):
                    break
            # start = 0
            # while start < len(segment):
            #     end = min(start + win_len, len(segment))
            #     sub_segment = segment[start:end]
            #     segmented_sample.append(sub_segment.values)
            #     start +=  hop_len
    elif segment_type == 'normal':
        for s_idx in range(0, len(sample), hop_len):
            e_idx = min(s_idx + win_len, len(sample))
            # s_idx_ = max(0, len(sample) - win_len) if e_idx == len(sample) else s_idx  # added: 07/07
            if (e_idx - s_idx) < 20:
                print('Warning: encounter too short segment with length less than 20.')
            segment = sample.iloc[s_idx:e_idx]
            segmented_sample.append(segment)
            if e_idx == len(sample):
                break
    else:
        raise Exception(f'Not supported segment type "{segment_type}" to segment.')
    return segmented_sample


# video id (ex, '23') <--> data partition (ex, 'train')
def get_data_partition(partition_file):
    vid2partition, partition2vid = {}, {}
    df = pd.read_csv(partition_file)

    for row in df.values:
        vid, partition = str(row[0]), row[1] # video id is string
        vid2partition[vid] = partition
        if partition not in partition2vid:
            partition2vid[partition] = []
        if vid not in partition2vid[partition]: # Note: this is necessary because few items repeat 2 times in partition file.
            partition2vid[partition].append(vid)

    return vid2partition, partition2vid


def get_padding_mask(x, x_lens):
    """
    :param x: (seq_len, batch_size, feature_dim)
    :param x_lens: sequence lengths within a batch with size (batch_size,)
    :return: padding_mask with size (batch_size, seq_len)
    """
    seq_len, batch_size, _ = x.size()
    mask = torch.ones(batch_size, seq_len, device=x.device)
    for seq, seq_len in enumerate(x_lens):
        mask[seq, :seq_len] = 0
    mask = mask.bool()
    return mask


class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, y_pred, y_true, seq_lens=None, label_smooth=None):
        """
        :param y_pred: (batch_size, seq_len)
        :param y_true: (batch_size, seq_len)
        :param seq_lens: (batch_size,)
        :return:
        """
        # make padding mask
        if seq_lens is not None:
            mask = torch.ones_like(y_true, device=y_true.device)
            for i, seq_len in enumerate(seq_lens):
                mask[i, seq_len:] = 0
        else:
            mask = torch.ones_like(y_true, device=y_true.device)
        # smooth label by average pooling
        if label_smooth is not None:
            y_true = torch.nn.functional.avg_pool1d(y_true.unsqueeze(1), kernel_size=label_smooth,
                                                    stride=1, padding=(label_smooth - 1) // 2,
                                                    count_include_pad=False)
            y_true = y_true.squeeze(1)

        y_true_mean = torch.sum(y_true * mask, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)
        y_pred_mean = torch.sum(y_pred * mask, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)
        # biased variance
        y_true_var = torch.sum(mask * (y_true - y_true_mean)**2, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)
        y_pred_var = torch.sum(mask * (y_pred - y_pred_mean)**2, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)

        cov = torch.sum(mask * (y_true - y_true_mean) * (y_pred - y_pred_mean), dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)

        ccc = torch.mean(2.0 * cov / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean)**2), dim=0) # (1,*)
        ccc = ccc.squeeze(0) # (*,) if necessary
        ccc_loss = 1.0 - ccc

        return ccc_loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true, seq_lens=None, label_smooth=None):
        """
        :param y_pred: (batch_size, seq_len)
        :param y_true: (batch_size, seq_len)
        :return:
        """
        # smooth label by average pooling
        if label_smooth is not None:
            y_true = torch.nn.functional.avg_pool1d(y_true.unsqueeze(1), kernel_size=label_smooth,
                                                    stride=1, padding=(label_smooth - 1) // 2,
                                                    count_include_pad=False)
            y_true = y_true.squeeze(1)

        # get mask
        if seq_lens is not None:
            mask = torch.ones_like(y_true, device=y_true.device)
            for i, seq_len in enumerate(seq_lens):
                mask[i, seq_len:] = 0
            loss = torch.nn.functional.mse_loss(y_pred, y_true, reduction='none')
            # loss = loss * mask
            # loss = loss.sum() / seq_lens.sum()
            mask = mask.bool()
            loss = loss.masked_select(mask)
            loss = loss.mean()
        else:
            loss = torch.nn.functional.mse_loss(y_pred, y_true)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, y_pred, y_true, seq_lens=None, label_smooth=None):
        """
        :param y_pred: (batch_size, seq_len)
        :param y_true: (batch_size, seq_len)
        :return:
        """
        # smooth label by average pooling
        if label_smooth is not None:
            y_true = torch.nn.functional.avg_pool1d(y_true.unsqueeze(1), kernel_size=label_smooth,
                                                    stride=1, padding=(label_smooth - 1) // 2,
                                                    count_include_pad=False)
            y_true = y_true.squeeze(1)

        # get mask
        if seq_lens is not None:
            mask = torch.ones_like(y_true, device=y_true.device)
            for i, seq_len in enumerate(seq_lens):
                mask[i, seq_len:] = 0
            loss = torch.nn.functional.l1_loss(y_pred, y_true, reduction='none')
            # loss = loss * mask
            # loss = loss.sum() / seq_lens.sum()
            mask = mask.bool()
            loss = loss.masked_select(mask)
            loss = loss.mean()
        else:
            loss = torch.nn.functional.l1_loss(y_pred, y_true)
        return loss


def eval(full_preds, full_labels):
    full_preds = np.row_stack(full_preds)
    full_labels = np.row_stack(full_labels)
    assert full_preds.shape == full_labels.shape
    n_targets = full_preds.shape[1]
    val_ccc, val_pcc, val_rmse = [], [], []
    for i in range(n_targets):
        preds = full_preds[:, i]
        labels = full_labels[:, i]
        ccc, pcc, rmse = cal_eval_metrics(preds, labels)
        val_ccc.append(ccc)
        val_pcc.append(pcc)
        val_rmse.append(rmse)
    return val_ccc, val_pcc, val_rmse


# ccc, pcc, mse
def cal_eval_metrics(preds, labels):
    rmse = np.sqrt(np.mean((preds - labels) ** 2))

    preds_mean, labels_mean = np.mean(preds), np.mean(labels)
    cov_mat = np.cov(preds, labels) # Note: unbiased
    covariance = cov_mat[0,1]
    preds_var, labels_var = cov_mat[0,0], cov_mat[1,1]

    pcc = covariance / np.sqrt(preds_var * labels_var)
    ccc = 2.0 * covariance / (preds_var + labels_var + (preds_mean - labels_mean)**2)
    return ccc, pcc, rmse


def save_model(model, params):
    model_dir = os.path.join(config.MODEL_FOLDER)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_file_name = f'{os.path.splitext(params.log_file_name)[0]}_[{params.n_seeds}_{params.current_seed}].pth'
    model_file = os.path.join(model_dir, model_file_name)
    torch.save(model, model_file)

    return model_file


def delete_model(model_file):
    if  os.path.exists(model_file):
        os.remove(model_file)
        print(f'Delete model "{model_file}".')
    else:
        print(f'Warning: model file "{model_file}" does not exist when delete it!')


def write_model_prediction(metas, preds, params, partition, view=False):
    """
    :param metas: # video id, time stamp, segment id
    :param preds:
    :param params:
    :param partition:
    :param view: whether plot predicted arousal and valence or not
    :return:
    """
    # write prediction sample by sample (multiple files)
    if params.save_dir is None:
        dir_name = f'{os.path.splitext(params.log_file_name)[0]}_[{params.n_seeds}_{params.current_seed}]'
        save_dir = os.path.join(config.PREDICTION_FOLDER, dir_name)
    else:
        save_dir = params.save_dir
    csv_dir = os.path.join(save_dir, 'csv')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    if view == True:
        img_dir = os.path.join(save_dir, 'img')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

    for idx, emo_dim in enumerate(params.emo_dim_set):
        csv_emo_dir = os.path.join(csv_dir, emo_dim)
        if not os.path.exists(csv_emo_dir):
            os.mkdir(csv_emo_dir)
        columns = ['timestamp', 'value', 'segment_id']
        for meta, pred in zip(metas, preds):
            vid = meta[0, 0]
            # csv
            sample_file_name = f'{vid}.csv'  # [vid].csv, ex: 1.csv
            sample_data = np.column_stack([meta[:,1], pred[:,idx], meta[:,2]])
            df = pd.DataFrame(sample_data, columns=columns)
            df[['timestamp', 'segment_id']] = df[['timestamp', 'segment_id']].astype(np.int)
            sample_pred_file = os.path.join(csv_emo_dir, sample_file_name)
            df.to_csv(sample_pred_file, index=False)

            # plot img
            if view == True:
                img_emo_dir = os.path.join(img_dir, emo_dim)
                if not os.path.exists(img_emo_dir):
                    os.mkdir(img_emo_dir)
                plot_video_prediction(df, partition, vid, emo_dim, img_emo_dir)

    # write aggregated prediction (all in one file)
    metas = np.row_stack(metas)
    metas = metas[:,:2]
    preds = np.row_stack(preds)
    data = np.column_stack([metas, preds])
    columns = ['id', 'timestamp'] + ['prediction_' + emo_dim for emo_dim in params.emo_dim_set]
    df = pd.DataFrame(data, columns=columns)
    df[['id', 'timestamp']] = df[['id', 'timestamp']].astype(np.int)
    pred_file_name = f'{partition}.csv'
    aggr_dir = os.path.join(csv_dir, 'aggregated')
    if not os.path.exists(aggr_dir):
        os.mkdir(aggr_dir)
    pred_file = os.path.join(aggr_dir, pred_file_name)
    if os.path.exists(pred_file):
        df_existed = pd.read_csv(pred_file)
        cols_existed = list(df_existed)
        cols = list(df)
        assert  len(cols) == 3 and len(cols_existed) == 3 and (cols[-1] != cols_existed[-1]), \
            f'Error: cannot merge existed prediction file "{pred_file}".'
        df = pd.merge(df, df_existed) if cols[-1] == 'prediction_arousal' else pd.merge(df_existed, df)
    df.to_csv(pred_file, index=False)


def plot_video_prediction(df_pred, partition, vid, emo_dim, save_dir):
    TIME_COLUMN = 'timestamp'
    EMO_COLUMN = 'value'

    label_file = os.path.join(config.PATH_TO_LABELS, emo_dim, f'{vid}.csv')
    df_label = pd.read_csv(label_file)

    time = df_pred[TIME_COLUMN].values / 1000.0 # ms --> s
    pred = df_pred[EMO_COLUMN].values
    if partition != 'test':
        label = df_label[EMO_COLUMN].values
    else:
        label = None
    # plot
    plt.figure(figsize=(20, 10))
    # color = 'r' if emo_dim == 'arousal' else 'g'
    plt.plot(time, pred, 'r-.', label=f'{emo_dim}(pred)')
    if label is not None:
        plt.plot(time, label, 'b', label=f'{emo_dim}(gt)')
    plt.title(f"{emo_dim} of Video '{vid}'")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    # set margin on x axis
    ax = plt.gca()
    if time[-1] < 400:
        x_interval = 10
    elif time[-1] < 800:
        x_interval = 20
    else:
        x_interval = 50
    x_major_locator = plt.MultipleLocator(x_interval)
    ax.xaxis.set_major_locator(x_major_locator)
    #y_major_locator = plt.MultipleLocator(0.2)
    #ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim([-1, 1])
    plt.grid()

    plt.savefig(os.path.join(save_dir, f'{vid}.jpg'))
    plt.close()


def write_fusion_result(metas, preds, params, partition, view=False):
    """
    :param metas: # video id, time stamp, segment id
    :param preds:
    :param params:
    :param partition:
    :param view: whether plot predicted arousal and valence or not
    :return:
    """
    # write prediction sample by sample (multiple files)
    if params.model == 'rnn':
        dir_name = f'{os.path.splitext(params.log_file_name)[0]}_[{params.n_seeds}_{params.current_seed}]'
    else: # machine learning model
        dir_name = f'{os.path.splitext(params.log_file_name)[0]}'
    csv_dir = os.path.join(params.base_dir, 'result', dir_name, 'csv')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    if view == True:
        img_dir = os.path.join(params.base_dir, 'result', dir_name, 'img')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

    for idx, emo_dim in enumerate(params.emo_dim_set):
        emo_dim_dir = os.path.join(csv_dir, emo_dim)
        if not os.path.exists(emo_dim_dir):
            os.mkdir(emo_dim_dir)
        columns = ['timestamp', 'value', 'segment_id']
        for meta, pred in zip(metas, preds):
            vid = meta[0, 0]
            # csv
            sample_file_name = f'{vid}.csv'  # [vid].csv, ex: 1.csv
            sample_data = np.column_stack([meta[:,1], pred[:,idx], meta[:,2]])
            df = pd.DataFrame(sample_data, columns=columns)
            df[['timestamp', 'segment_id']] = df[['timestamp', 'segment_id']].astype(np.int)
            sample_pred_file = os.path.join(emo_dim_dir, sample_file_name)
            df.to_csv(sample_pred_file, index=False)

            # plot img
            if view == True:
                save_dir = os.path.join(img_dir, emo_dim)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                plot_video_prediction(df, partition, vid, emo_dim, save_dir)

    # write aggregated prediction (all in one file)
    metas = np.row_stack(metas)
    metas = metas[:,:2]
    preds = np.row_stack(preds)
    data = np.column_stack([metas, preds])
    columns = ['id', 'timestamp'] + ['prediction_' + emo_dim for emo_dim in params.emo_dim_set]
    df = pd.DataFrame(data, columns=columns)
    df[['id', 'timestamp']] = df[['id', 'timestamp']].astype(np.int)
    pred_file_name = f'{partition}.csv'
    aggr_dir = os.path.join(csv_dir, 'aggregated')
    if not os.path.exists(aggr_dir):
        os.mkdir(aggr_dir)
    pred_file = os.path.join(aggr_dir, pred_file_name)
    df.to_csv(pred_file, index=False)