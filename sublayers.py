''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        mask1 = mask.unsqueeze(-2)
        attn = attn.masked_fill(mask1 == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))

        output = torch.matmul(attn, v)
        mask2 = mask.unsqueeze(-1)
        output = output.masked_fill(mask2 == 0, 0)
        return output

class SelfAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.w1 = nn.Linear(d_model, d_model, bias=False)
        self.w2 = nn.Linear(d_model, d_model, bias=False)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, seq_cur, seq_total, mask_cur, mask_total, reverse=False):
        batch_size, n_blocks, total_len = seq_total.size(0), seq_total.size(1), seq_total.size(2)
        cur_len = seq_cur.size(2)

        k = self.act(self.w1(seq_cur)).view(batch_size*n_blocks, cur_len, -1)
        v = self.act(self.w2(seq_total)).view(batch_size*n_blocks, total_len, -1).transpose(1,2).contiguous()
        mask_total = mask_total.view(batch_size*n_blocks, total_len)
        attn = torch.matmul(k / (self.d_model ** 0.5), v)
        mask_total = mask_total.unsqueeze(1)
        attn = attn.masked_fill(mask_total == 0, 0.0)
        attn_cur = torch.sum(attn, -1)
        if reverse:
            attn_cur = -attn_cur
        mask_cur = mask_cur.view(batch_size*n_blocks, cur_len)
        attn_cur = attn_cur.masked_fill(mask_cur == 0, -1e9)
        attn_cur = F.softmax(attn_cur, dim=-1).view(batch_size, n_blocks, cur_len)
        return attn_cur


class AttendAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.w1 = nn.Linear(d_model, d_model, bias=False)
        self.w2 = nn.Linear(d_model, d_model, bias=False)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, mask, att_vector):
        batch_size, seqlen = seq.size(0), seq.size(1)
        q = seq.view(batch_size, seqlen, -1)
        k = self.act(self.w1(att_vector)).view(batch_size, 1, -1)
        v = self.act(self.w2(seq)).view(batch_size, seqlen, -1).transpose(1,2).contiguous()
        mask = mask.view(batch_size, seqlen)
        attn = torch.matmul(k / (self.d_model ** 0.5), v).squeeze(1)
        attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        return attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None, mask_q=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # if mask is not None:
        #     mask = mask.unsqueeze(-1)   # For head axis broadcasting.
        q = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)

        x += residual
        x = self.layer_norm(x)
        return x

class AttentiveKernelPooling(nn.Module):
    ''' Kernel pooling module '''

    def __init__(self):
        super(AttentiveKernelPooling, self).__init__()
        mu = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9]
        sigma = [1e-3] + [0.1] * 10
        self.n_bins = len(mu)
        tensor_mu = torch.FloatTensor(mu)
        tensor_sigma = torch.FloatTensor(sigma)
        tensor_mu = tensor_mu.cuda()
        tensor_sigma = tensor_sigma.cuda()

        self.mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, self.n_bins)
        self.sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, self.n_bins)
        self.classifier = nn.Linear(self.n_bins, 1, 1)

    def forward(self, matrix, mask_row, mask_col, attention):
        batch_size, row, col = matrix.size(0), matrix.size(1), matrix.size(2)
        mask_col = mask_col.unsqueeze(1)
        pooling_value = torch.exp((- ((matrix - self.mu) ** 2) / (self.sigma ** 2) / 2))*mask_col.unsqueeze(-1)
        pooling_sum = torch.sum(pooling_value, -2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01 * mask_row.unsqueeze(-1)
        if attention != None:
            attention = attention.unsqueeze(-1)
            mu_attention = self.mu.squeeze(2)
            sigma_attention = self.sigma.squeeze(2)
            attention = torch.exp((- ((attention - self.mu) ** 2) / (self.sigma ** 2) / 2))
            mask_attention = mask_row.unsqueeze(-1).expand(batch_size, row, self.n_bins).contiguous()
            attention = attention.masked_fill(mask_attention == 0, -1e9)
            attention = F.softmax(attention, dim=1)
            log_pooling_sum = torch.sum(log_pooling_sum * attention, -2)
        else:
            log_pooling_sum = torch.sum(log_pooling_sum, -2)
        # log_pooling_sum = torch.sum(log_pooling_sum, -2)
        return log_pooling_sum


class KernelPooling(nn.Module):
    ''' Kernel pooling module '''

    def __init__(self):
        super(KernelPooling, self).__init__()
        mu = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9]
        sigma = [1e-3] + [0.1] * 10
        self.n_bins = len(mu)
        tensor_mu = torch.FloatTensor(mu)
        tensor_sigma = torch.FloatTensor(sigma)
        tensor_mu = tensor_mu.cuda()
        tensor_sigma = tensor_sigma.cuda()

        self.mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, self.n_bins)
        self.sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, self.n_bins)
        self.classifier = nn.Linear(self.n_bins, 1, 1)

    def forward(self, matrix, mask_row, mask_col):
        batch_size, row, col = matrix.size(0), matrix.size(1), matrix.size(2)
        mask_col = mask_col.unsqueeze(1)
        pooling_value = torch.exp((- ((matrix - self.mu) ** 2) / (self.sigma ** 2) / 2))*mask_col.unsqueeze(-1)
        pooling_sum = torch.sum(pooling_value, -2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01 * mask_row.unsqueeze(-1)
        log_pooling_sum = torch.sum(log_pooling_sum, -2)
        # log_pooling_sum = torch.squeeze(F.tanh(self.classifier(log_pooling_sum)), -1)
        # log_pooling_sum = self.classifier(log_pooling_sum)
        return log_pooling_sum


class ConvKernelPooling(nn.Module):
    ''' Kernel pooling module '''

    def __init__(self):
        super(ConvKernelPooling, self).__init__()
        mu = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9]
        sigma = [1e-3] + [0.1] * 10
        self.n_bins = len(mu)
        tensor_mu = torch.FloatTensor(mu)
        tensor_sigma = torch.FloatTensor(sigma)
        tensor_mu = tensor_mu.cuda()
        tensor_sigma = tensor_sigma.cuda()

        self.mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, self.n_bins)
        self.sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, self.n_bins)
        self.dense = nn.Linear(self.n_bins, 1, 1)

    def forward(self, matrix, mask_row, mask_col):
        matrix = matrix.unsqueeze(-1).contiguous()
        batch_size, row, col = matrix.size(0), matrix.size(1), matrix.size(2)
        mask_col = mask_col.unsqueeze(1)
        pooling_value = torch.exp((- ((matrix - self.mu) ** 2) / (self.sigma ** 2) / 2))*mask_col.unsqueeze(-1)
        pooling_sum = torch.sum(pooling_value, -2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01 * mask_row.unsqueeze(-1)
        log_pooling_sum = torch.sum(log_pooling_sum , -2)
        return log_pooling_sum


class ReformulationClassifier(nn.Module):
    def __init__(self, d_model, type_num, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.w1 = nn.Linear(d_model, d_model, bias=False)
        self.w2 = nn.Linear(d_model, d_model, bias=False)
        self.act = nn.LeakyReLU()
        self.classifier = nn.Linear(3, type_num, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, retained, removed, added, combine_mean, prevs_mean, nexts_mean):

        def compare(cmp1, cmp2):
            cmp1 = F.normalize(self.act(self.w1(cmp1)), p=2, dim=-1, eps=1e-10)
            cmp2 = F.normalize(self.act(self.w2(cmp2)), p=2, dim=-1, eps=1e-10)
            return torch.sum(cmp1*cmp2, -1)
        sim_retained = compare(retained, combine_mean)
        sim_removed = compare(removed, prevs_mean)
        sim_added = compare(added, nexts_mean)
        sim = torch.stack((sim_retained, sim_removed, sim_added),-1)
        logits = F.softmax(self.classifier(sim), -1)

        return logits