''' Define the Layers '''
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sublayers import KernelPooling, MultiHeadAttention, PositionwiseFeedForward, AttentiveKernelPooling, SelfAttention, AttendAttention

__author__ = "Yu-Hsiang Huang"

class EncodingLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout=0.1):
        super(EncodingLayer, self).__init__()
        self.d_model, self.heads = d_model, heads
        self.d_per_head = self.d_model // self.heads
        self.slf_attn = MultiHeadAttention(self.heads, self.d_model, self.d_per_head, self.d_per_head, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, word_vec, mask_local):
        batch_size, seq_len = word_vec.size(0), word_vec.size(1)
        mask_local = mask_local.view(batch_size, 1, seq_len)
        word_vec = word_vec * mask_local.transpose(1,2).contiguous()
        mask_local_expand = mask_local.expand(batch_size, self.heads, seq_len).contiguous()
        word_vec = self.slf_attn(word_vec, word_vec, word_vec, mask_local_expand)
        word_vec = self.feed_forward(word_vec)
        word_vec = word_vec * mask_local.transpose(1,2).contiguous()
        return word_vec

class ReformulationEncoder(nn.Module):
    def __init__(self, d_model, type_num, dropout=0.1):
        super(ReformulationEncoder, self).__init__()
        self.d_model = d_model
        self.type_num = type_num
        self.retained_atten_layer = SelfAttention(self.d_model)
        self.added_atten_layer = SelfAttention(self.d_model)
        self.removed_atten_layer = SelfAttention(self.d_model)

    def forward(self, cur, cur_mask, prev, prev_mask, nexts, nexts_mask):

        batch_size, n_blocks, n_tokens_q = cur.size(0), cur.size(1), cur.size(2)
        n_tokens_qd = prev.size(2)

        seq = torch.cat([cur, prev], 2)
        seq_mask = torch.cat([cur_mask, prev_mask], 2)
        retained_atten = self.retained_atten_layer(cur, seq, cur_mask, seq_mask)
        added_atten = self.added_atten_layer(cur, seq, cur_mask, seq_mask, True)

        seq = torch.cat([cur, nexts], 2)
        seq_mask = torch.cat([cur_mask, nexts_mask], 2)
        removed_atten = self.removed_atten_layer(cur, seq, cur_mask, seq_mask, True)

        return retained_atten, added_atten, removed_atten

class ReformulationRepresentation(nn.Module):
    def __init__(self):
        super(ReformulationRepresentation, self).__init__()

    def forward(self, word_vec_q_encode, mask_query, retained_atten, added_atten, removed_atten):

        retained_rep = torch.sum(word_vec_q_encode * retained_atten.unsqueeze(-1), 2)
        added_rep = torch.sum(word_vec_q_encode * added_atten.unsqueeze(-1), 2)
        removed_rep = torch.sum(word_vec_q_encode * removed_atten.unsqueeze(-1), 2)

        querylen = torch.sum(mask_query, -1).unsqueeze(-1) + 1e-4
        mean_rep = torch.sum(word_vec_q_encode, 2) / querylen

        return retained_rep, added_rep, removed_rep, mean_rep


class RepIneractionLayer(nn.Module):
    def __init__(self, d_model):
        super(RepIneractionLayer, self).__init__()
        self.trans = nn.Linear(d_model * 4, d_model)
        self.w1 = nn.Linear(d_model, d_model, bias=False)
        self.w2 = nn.Linear(d_model, d_model, bias=False)

    def forward(self, retained_rep_cur, added_rep_cur, removed_rep_cur, mean_rep_cur, can_rep):

        rep = torch.cat([retained_rep_cur, added_rep_cur, removed_rep_cur, mean_rep_cur], -1)
        rep_cur = F.tanh(self.trans(rep))

        can_rep_norm = F.normalize(F.tanh(self.w1(can_rep)), p=2, dim=-1, eps=1e-10)
        rep_cur_norm = F.normalize(F.tanh(self.w2(rep_cur)), p=2, dim=-1, eps=1e-10)
        score = torch.sum(can_rep_norm * rep_cur_norm, -1)

        return score


class ReformulationClassifier(nn.Module):
    def __init__(self, d_model, type_num, dropout=0.1):
        super(ReformulationClassifier, self).__init__()
        self.d_model = d_model
        self.type_num = type_num
        self.dense = nn.Linear(3 * d_model, type_num, 1)

    def forward(self, cq, cq_mask, lq, lq_mask, retained_atten, added_atten, removed_atten):

        batch_size, n_tokens_q = cq.size(0), cq.size(1)
        cq = cq * cq_mask.unsqueeze(-1)
        lq = lq * lq_mask.unsqueeze(-1)
        cq_mean = torch.sum(cq, 1) / (torch.sum(cq_mask, 1).unsqueeze(-1) + 1e-4)
        lq_mean = torch.sum(lq, 1) / (torch.sum(lq_mask, 1).unsqueeze(-1) + 1e-4)
        retained = torch.sum(cq * retained_atten.unsqueeze(-1), 1)
        added = torch.sum(cq * added_atten.unsqueeze(-1), 1)
        removed = torch.sum(lq * removed_atten.unsqueeze(-1), 1)

        combine = torch.cat([retained - cq_mean, added - cq_mean, removed - lq_mean], -1).view(batch_size, 3 * self.d_model)
        logits = F.tanh(self.dense(combine))

        return logits

class AttentiveKnrmLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AttentiveKnrmLayer, self).__init__()
        self.d_model = d_model
        self.n_bins = 11
        self.attentive_knrm_pooling = AttentiveKernelPooling()
        self.query_weighting = AttendAttention(d_model, dropout)
        self.dense = nn.Linear(self.n_bins, 1, 1)

    def forward(self, word_atten, hq, can, rep, rep_cur, mask_hq, mask_can, mask_session):

        batch_size, n_blocks, n_tokens_q = hq.size(0), hq.size(1), hq.size(2)
        n_tokens_d = can.size(1)

        can = can.unsqueeze(1).expand(batch_size, n_blocks, n_tokens_d, self.d_model).contiguous()


        can_norm = F.normalize(can, p=2, dim=-1, eps=1e-10).view(batch_size*n_blocks, n_tokens_d, -1)
        hq_norm = F.normalize(hq, p=2, dim=-1, eps=1e-10).view(batch_size*n_blocks, n_tokens_q, -1)
        matrix = torch.bmm(hq_norm, can_norm.transpose(1,2).contiguous()).view(batch_size * n_blocks, n_tokens_q, n_tokens_d, 1)

        mask_row = mask_hq.view(batch_size * n_blocks, n_tokens_q)
        mask_col = mask_can.unsqueeze(1).expand(batch_size, n_blocks, n_tokens_d).contiguous().view(batch_size * n_blocks, n_tokens_d)
        if word_atten != None:
            attention = word_atten.view(batch_size * n_blocks, n_tokens_q)
        else:
            attention = None

        pooled_can = self.attentive_knrm_pooling(matrix, mask_row, mask_col, attention).view(batch_size, n_blocks, -1)

        pooled_can = pooled_can * mask_session.unsqueeze(-1)


        attn = self.query_weighting(rep, mask_session, rep_cur)
        # pooled_cq = torch.sum(pooled_can * attn.unsqueeze(-1), 1).view(batch_size, -1)
        score = F.tanh(self.dense(pooled_can)).squeeze(-1)
        # sl = torch.sum(mask_session, 1) + 1e-4
        # score = torch.sum(score, 1) / sl
        score = torch.sum(score * attn, 1)

        return score


class AttentiveConvKnrmLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AttentiveConvKnrmLayer, self).__init__()
        self.d_model = d_model
        self.n_bins = 11

        self.conv_uni = nn.Sequential(
            nn.Conv2d(1, 128, (1, d_model)),
            nn.ReLU()
        )

        self.conv_bi = nn.Sequential(
            nn.Conv2d(1, 128, (2, d_model), padding=(1,0)),
            nn.ReLU()
        )
        self.conv_tri = nn.Sequential(
            nn.Conv2d(1, 128, (3, d_model), padding=(1,0)),
            nn.ReLU()
        )

        self.attentive_knrm_pooling = AttentiveKernelPooling()
        self.query_weighting = AttendAttention(d_model, dropout)
        self.dense = nn.Linear(self.n_bins*9, 1, 1)

    def forward(self, word_atten, hq, can, rep, rep_cur, mask_hq, mask_can, mask_session):

        batch_size, n_blocks, n_tokens_q = hq.size(0), hq.size(1), hq.size(2)
        n_tokens_d = can.size(1)

        d = can.unsqueeze(1).expand(batch_size, n_blocks, n_tokens_d, self.d_model).contiguous().view(batch_size*n_blocks, 1, n_tokens_d, -1)
        q = hq.view(batch_size * n_blocks, 1, n_tokens_q, -1)

        qu = torch.transpose(torch.squeeze(self.conv_uni(q)), 1, 2).contiguous()
        qb = torch.transpose(torch.squeeze(self.conv_bi(q)), 1, 2).contiguous()
        qt = torch.transpose(torch.squeeze(self.conv_tri(q)), 1, 2).contiguous()
        qb = qb[:,:-1,:]

        du = torch.squeeze(self.conv_uni(d))
        db = torch.squeeze(self.conv_bi(d))
        dt = torch.squeeze(self.conv_tri(d))
        db = db[:,:,:-1]

        qu_norm = F.normalize(qu, p=2, dim=-1, eps=1e-10)
        qb_norm = F.normalize(qb, p=2, dim=-1, eps=1e-10)
        qt_norm = F.normalize(qt, p=2, dim=-1, eps=1e-10)
        du_norm = F.normalize(du, p=2, dim=-2, eps=1e-10)
        db_norm = F.normalize(db, p=2, dim=-2, eps=1e-10)
        dt_norm = F.normalize(dt, p=2, dim=-2, eps=1e-10)

        # matrix = torch.matmul(hq_norm, can_norm.transpose(2,3).contiguous()).view(batch_size * n_blocks, n_tokens_q, n_tokens_d)
        mask_row = mask_hq.view(batch_size * n_blocks, n_tokens_q)
        mask_col = mask_can.unsqueeze(1).expand(batch_size, n_blocks, n_tokens_d).contiguous().view(batch_size * n_blocks, n_tokens_d)
        if word_atten != None:
            attention = word_atten.view(batch_size * n_blocks, n_tokens_q)
        else:
            attention = None


        k_uu = self.attentive_knrm_pooling(torch.bmm(qu_norm, du_norm).view(batch_size * n_blocks, n_tokens_q, n_tokens_d), mask_row, mask_col, attention).view(batch_size, n_blocks, -1) * mask_session.unsqueeze(-1)
        k_ub = self.attentive_knrm_pooling(torch.bmm(qu_norm, db_norm).view(batch_size * n_blocks, n_tokens_q, n_tokens_d), mask_row, mask_col, attention).view(batch_size, n_blocks, -1) * mask_session.unsqueeze(-1)
        k_ut = self.attentive_knrm_pooling(torch.bmm(qu_norm, dt_norm).view(batch_size * n_blocks, n_tokens_q, n_tokens_d), mask_row, mask_col, attention).view(batch_size, n_blocks, -1) * mask_session.unsqueeze(-1)
        k_bu = self.attentive_knrm_pooling(torch.bmm(qb_norm, du_norm).view(batch_size * n_blocks, n_tokens_q, n_tokens_d), mask_row, mask_col, attention).view(batch_size, n_blocks, -1) * mask_session.unsqueeze(-1)
        k_bb = self.attentive_knrm_pooling(torch.bmm(qb_norm, db_norm).view(batch_size * n_blocks, n_tokens_q, n_tokens_d), mask_row, mask_col, attention).view(batch_size, n_blocks, -1) * mask_session.unsqueeze(-1)
        k_bt = self.attentive_knrm_pooling(torch.bmm(qb_norm, dt_norm).view(batch_size * n_blocks, n_tokens_q, n_tokens_d), mask_row, mask_col, attention).view(batch_size, n_blocks, -1) * mask_session.unsqueeze(-1)
        k_tu = self.attentive_knrm_pooling(torch.bmm(qt_norm, du_norm).view(batch_size * n_blocks, n_tokens_q, n_tokens_d), mask_row, mask_col, attention).view(batch_size, n_blocks, -1) * mask_session.unsqueeze(-1)
        k_tb = self.attentive_knrm_pooling(torch.bmm(qt_norm, db_norm).view(batch_size * n_blocks, n_tokens_q, n_tokens_d), mask_row, mask_col, attention).view(batch_size, n_blocks, -1) * mask_session.unsqueeze(-1)
        k_tt = self.attentive_knrm_pooling(torch.bmm(qt_norm, dt_norm).view(batch_size * n_blocks, n_tokens_q, n_tokens_d), mask_row, mask_col, attention).view(batch_size, n_blocks, -1) * mask_session.unsqueeze(-1)

        pooled_can = torch.cat([k_uu, k_ut, k_ub, k_bu, k_tu, k_bb, k_bt, k_tb, k_tt], -1)

        attn = self.query_weighting(rep, mask_session, rep_cur)

        pooled_cq = torch.sum(pooled_can * attn.unsqueeze(-1), 1).view(batch_size, -1)
        score = F.tanh(self.dense(pooled_cq)).squeeze(-1)

        return score

class KNRMLayer(nn.Module):
    def __init__(self, d_model):
        super(KNRMLayer, self).__init__()
        self.d_model = d_model
        self.n_bins = 11
        self.knrm_pooling = KernelPooling()
        self.classifier = nn.Linear(self.n_bins, 1, 1)

    def forward(self, current_query, candidate, current_query_mask, candidate_mask):
    # def forward(self, matrix, current_query_mask, candidate_mask):


        batch_size, n_tokens_q = current_query.size(0), current_query.size(1)
        n_tokens_d = candidate.size(1)

        can_norm = F.normalize(candidate, p=2, dim=-1, eps=1e-10).transpose(1,2).contiguous()
        cq_norm = F.normalize(current_query, p=2, dim=-1, eps=1e-10)
        matrix = torch.bmm(cq_norm, can_norm).view(batch_size, n_tokens_q, n_tokens_d, 1)

        feat = self.knrm_pooling(matrix, current_query_mask, candidate_mask)
        score = torch.squeeze(F.tanh(self.classifier(feat)), -1)

        return score

class ConvKNRMLayer(nn.Module):
    def __init__(self, d_model):
        super(ConvKNRMLayer, self).__init__()
        self.d_model = d_model
        self.n_bins = 11
        self.conv_uni = nn.Sequential(
            nn.Conv2d(1, 128, (1, d_model)),
            nn.ReLU()
        )

        self.conv_bi = nn.Sequential(
            nn.Conv2d(1, 128, (2, d_model), padding=(1,0)),
            nn.ReLU()
        )
        self.conv_tri = nn.Sequential(
            nn.Conv2d(1, 128, (3, d_model), padding=(1,0)),
            nn.ReLU()
        )
        self.knrm_pooling = ConvKernelPooling()
        self.dense = nn.Linear(self.n_bins*9, 1, 1)

    def forward(self, current_query, candidate, current_query_mask, candidate_mask):

        batch_size, n_tokens_q = current_query.size(0), current_query.size(1)
        n_tokens_d = candidate.size(1)
        q = current_query.view(batch_size, 1, n_tokens_q, -1)
        d = candidate.view(batch_size, 1, n_tokens_d, -1)

        qu = torch.transpose(torch.squeeze(self.conv_uni(q)), 1, 2).contiguous()
        qb = torch.transpose(torch.squeeze(self.conv_bi(q)), 1, 2).contiguous()
        qt = torch.transpose(torch.squeeze(self.conv_tri(q)), 1, 2).contiguous()
        qb = qb[:,:-1,:]

        du = torch.squeeze(self.conv_uni(d))
        db = torch.squeeze(self.conv_bi(d))
        dt = torch.squeeze(self.conv_tri(d))
        db = db[:,:,:-1]

        qu_norm = F.normalize(qu, p=2, dim=-1, eps=1e-10)
        qb_norm = F.normalize(qb, p=2, dim=-1, eps=1e-10)
        qt_norm = F.normalize(qt, p=2, dim=-1, eps=1e-10)
        du_norm = F.normalize(du, p=2, dim=-2, eps=1e-10)
        db_norm = F.normalize(db, p=2, dim=-2, eps=1e-10)
        dt_norm = F.normalize(dt, p=2, dim=-2, eps=1e-10)

        # matrix = torch.bmm(qu_norm, du_norm).view(batch_size, n_tokens_q, n_tokens_d)
        # matrix = torch.matmul(cq_norm, can_norm.transpose(1,2).contiguous()).view(batch_size, n_tokens_q, n_tokens_d)
        mask_row = current_query_mask
        mask_col = candidate_mask
        k_uu = self.knrm_pooling(torch.bmm(qu_norm, du_norm).view(batch_size, n_tokens_q, n_tokens_d), mask_row, mask_col).squeeze(-1)
        k_ub = self.knrm_pooling(torch.bmm(qu_norm, db_norm).view(batch_size, n_tokens_q, n_tokens_d), mask_row, mask_col).squeeze(-1)
        k_ut = self.knrm_pooling(torch.bmm(qu_norm, dt_norm).view(batch_size, n_tokens_q, n_tokens_d), mask_row, mask_col).squeeze(-1)
        k_bu = self.knrm_pooling(torch.bmm(qb_norm, du_norm).view(batch_size, n_tokens_q, n_tokens_d), mask_row, mask_col).squeeze(-1)
        k_bb = self.knrm_pooling(torch.bmm(qb_norm, db_norm).view(batch_size, n_tokens_q, n_tokens_d), mask_row, mask_col).squeeze(-1)
        k_bt = self.knrm_pooling(torch.bmm(qb_norm, dt_norm).view(batch_size, n_tokens_q, n_tokens_d), mask_row, mask_col).squeeze(-1)
        k_tu = self.knrm_pooling(torch.bmm(qt_norm, du_norm).view(batch_size, n_tokens_q, n_tokens_d), mask_row, mask_col).squeeze(-1)
        k_tb = self.knrm_pooling(torch.bmm(qt_norm, db_norm).view(batch_size, n_tokens_q, n_tokens_d), mask_row, mask_col).squeeze(-1)
        k_tt = self.knrm_pooling(torch.bmm(qt_norm, dt_norm).view(batch_size, n_tokens_q, n_tokens_d), mask_row, mask_col).squeeze(-1)
        pooled_can = torch.cat([k_uu, k_ut, k_ub, k_bu, k_tu, k_bb, k_bt, k_tb, k_tt], -1)
        score = F.tanh(self.dense(pooled_can)).squeeze(-1)

        return score



class GlobalIntentEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(GlobalIntentEncoder, self).__init__()
        self.d_model = d_model
        self.removed_attn = SelfCombinedAttention(self.d_model, dropout=dropout)
        self.retained_attn = MaskedAttendAttention(self.d_model, dropout=dropout)
        self.added_attn = MaskedAttendAttention(self.d_model, dropout=dropout)

    def forward(self, retained, removed, added, mask_session, mask_matrix):
        batch_size, n_blocks = retained.size(0), retained.size(1)
        g_removed = self.removed_attn(removed.unsqueeze(1), mask_session).squeeze(1)
        g_retained = self.retained_attn(retained, removed, mask_session, mask_matrix)
        g_added = self.added_attn(added, removed, mask_session, mask_matrix)
        return g_removed, g_retained, g_added

class Ranker(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Ranker, self).__init__()
        self.d_model = d_model
        self.retained_ranker = AttentiveRanker(self.d_model, dropout=dropout)
        self.removed_ranker = AttentiveRanker(self.d_model, dropout=dropout)
        self.added_ranker = AttentiveRanker(self.d_model, dropout=dropout)
        self.classifier = nn.Linear(6, 2, bias=False)

    def forward(self, retained_q, retained_d, removed_q, removed_d, added_q, added_d, candidate, mask):
        batch_size, seq_len = candidate.size(0), candidate.size(1)
        sim_ret_q = self.retained_ranker(candidate, mask, retained_q)
        sim_ret_d = self.retained_ranker(candidate, mask, retained_d)

        sim_rem_q = self.removed_ranker(candidate, mask, retained_q)
        sim_rem_d = self.removed_ranker(candidate, mask, retained_d)

        sim_add_q = self.added_ranker(candidate, mask, retained_q)
        sim_add_d = self.added_ranker(candidate, mask, retained_d)

        sim = torch.stack((sim_ret_q, sim_ret_d, sim_rem_q, sim_rem_d, sim_add_q, sim_add_d), -1)
        logits = F.softmax(self.classifier(sim), dim=-1)
        return logits
