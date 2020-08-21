# *_*coding:utf-8 *_*
import copy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import Module, ModuleList
from torch.nn import Dropout, LayerNorm
from torch.nn import MultiheadAttention

import utils


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class SelfAttentionLayer(Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super(SelfAttentionLayer, self).__init__()
        self.multihead_attn = MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.multihead_attn(src, src, src, attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        return src


class SelfAttention(Module):
    def __init__(self, n_layers, d_model, n_heads, dropout=0.0):
        super(SelfAttention, self).__init__()
        self.n_layers = n_layers
        self.layers = _get_clones(SelfAttentionLayer(d_model, n_heads, dropout), n_layers)

    def forward(self, x, x_padding_mask):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=x_padding_mask)
        return x


class RNNEncoder(nn.Module):
    def __init__(self, d_in, d_out, n_layers=1, bi=True, dropout=0.2, type='gru'):
        super(RNNEncoder, self).__init__()
        self.type = type.lower()
        if self.type == 'gru':
            self.rnn = nn.GRU(input_size=d_in, hidden_size=d_out,
                          bidirectional=bi, num_layers=n_layers, dropout=dropout)
        elif self.type == 'lstm':
            self.rnn = nn.LSTM(input_size=d_in, hidden_size=d_out,
                            bidirectional=bi, num_layers=n_layers, dropout=dropout)
        else:
            raise Exception(f'Not supported rnn type ("{type}")!')

    def forward(self, x, x_len):
        """
        :param x: torch tensor, (batch_size, seq_len, feature_dim). Note: batch first
        :param x_len: torch tensor, (batch_size,)
        :return:
        """
        # sort by length
        x_sorted_len, idx_sort = torch.sort(x_len, dim=0, descending=True)
        idx_unsort = torch.argsort(idx_sort)
        x_sorted = x.index_select(0, idx_sort)

        # pack
        x_packed = pack_padded_sequence(x_sorted, x_sorted_len, batch_first=True)
        x_transformed = self.rnn(x_packed)[0]
        # pad
        x_padded = pad_packed_sequence(x_transformed, total_length=x.size(1), batch_first=True)[0]

        # unsort by length
        x_padded = x_padded.index_select(0, idx_unsort)

        return x_padded


class Regressor(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0, bias=0):
        super(Regressor, self).__init__()
        self.fc_1 = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(True), nn.Dropout(dropout))
        self.fc_2 = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.fc_2.bias.data, bias)

    def forward(self, x):
        y = self.fc_2(self.fc_1(x))
        return y


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        if params.d_in != params.d_rnn:
            self.proj = nn.Linear(params.d_in, params.d_rnn, bias=False)
            d_rnn_in = params.d_rnn
        else:
            d_rnn_in = params.d_in

        if params.attn == True:
            self.attn = SelfAttention(params.n_layers, d_rnn_in, params.n_heads,dropout=params.attn_dr)

        if params.rnn_n_layers > 0:
            self.rnn = RNNEncoder(d_rnn_in, params.d_rnn, n_layers=params.rnn_n_layers,
                                    bi=params.rnn_bi, dropout=params.rnn_dr, type=params.rnn)
            d_rnn_out = params.d_rnn * 2 if params.rnn_bi else params.d_rnn
        else:
            d_rnn_out = params.d_rnn

        self.out = Regressor(d_rnn_out, params.d_out_fc,  len(params.emo_dim_set), dropout=params.out_dr)

    def forward(self, x, x_len):
        if self.params.d_in != self.params.d_rnn:
            x = self.proj(x)

        if self.params.attn == True:
            x = x.transpose(0, 1) # (seq_len, batch_size, feature_dim)
            mask = utils.get_padding_mask(x, x_len)
            x = self.attn(x, mask)
            x = x.transpose(0, 1) # (batch_size, seq_len, feature_dim)

        if self.params.rnn_n_layers > 0:
            x = self.rnn(x, x_len)

        y = self.out(x)
        return y


class FusionModel(nn.Module):
    def __init__(self, params):
        super(FusionModel, self).__init__()
        self.params = params

        self.rnn = RNNEncoder(params.d_in, params.d_model, n_layers=params.n_layers,
                              bi=params.rnn_bi, dropout=params.dr, type=params.rnn)

        d_rnn_out = params.d_model * 2 if params.rnn_bi else params.d_model

        self.out = nn.Linear(d_rnn_out, params.d_out)

    def forward(self, x, x_len):
        x = self.rnn(x, x_len)
        y = self.out(x)
        return y