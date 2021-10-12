''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from layers import EncodingLayer, ReformulationEncoder, RepIneractionLayer, AttentiveKnrmLayer, KNRMLayer, ReformulationClassifier
# from eagle.layers import *
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import math

__author__ = "Xiaochen Zuo"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=200):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class HQCN(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, vocab, d_word_vec=100, d_model=100, d_inner=200,
            n_layers=1, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200):

        super().__init__()

        self.pad_idx = vocab.get_id(vocab.pad_token)
        self.d_word_vec = d_word_vec
        self.num_layers = n_layers
        self.class_weight_reform = torch.FloatTensor([35378, 298660, 598312, 267352]).cuda() # AOL
        # self.class_weight_reform = torch.FloatTensor([36575, 281827, 788722, 261177]).cuda() # Tiangong
        # 归一+取倒数+归一
        self.class_weight_reform = self.class_weight_reform.sum() / self.class_weight_reform
        self.class_weight_reform = self.class_weight_reform / self.class_weight_reform.sum()
        self.reform_type = self.class_weight_reform.size(0)

        self.word_emb = nn.Embedding(len(vocab.embeddings), self.d_word_vec, padding_idx = self.pad_idx)
        self.word_emb.weight.data.copy_(torch.from_numpy(vocab.embeddings))

        self.pos_emb = PositionalEncoding(dropout, d_word_vec, n_position)
        self.query_encoding_layers = nn.ModuleList(
            [EncodingLayer(d_model, d_inner, n_head, dropout=dropout) for i in range(n_layers)])

        self.document_encoding_layers = nn.ModuleList(
            [EncodingLayer(d_model, d_inner, n_head, dropout=dropout) for i in range(n_layers)])


        self.retained_encoding_layers = nn.ModuleList(
            [EncodingLayer(d_model, d_inner, n_head, dropout=dropout) for i in range(n_layers)])

        self.added_encoding_layers = nn.ModuleList(
            [EncodingLayer(d_model, d_inner, n_head, dropout=dropout) for i in range(n_layers)])

        self.removed_encoding_layers = nn.ModuleList(
            [EncodingLayer(d_model, d_inner, n_head, dropout=dropout) for i in range(n_layers)])

        self.mean_encoding_layers = nn.ModuleList(
            [EncodingLayer(d_model, d_inner, n_head, dropout=dropout) for i in range(n_layers)])

        self.reformulation_atten = ReformulationEncoder(d_model, self.reform_type, dropout=dropout)
        # self.reformulation_rep = ReformulationRepresentation()
        self.reformulation_classifier = ReformulationClassifier(d_model, self.reform_type)
        self.rep_interaction_layer = RepIneractionLayer(d_model)
        self.retained_attentive_knrm_layer = AttentiveKnrmLayer(d_model, dropout=dropout)
        self.added_attentive_knrm_layer = AttentiveKnrmLayer(d_model, dropout=dropout)
        self.removed_attentive_knrm_layer = AttentiveKnrmLayer(d_model, dropout=dropout)
        self.mean_attentive_knrm_layer = AttentiveKnrmLayer(d_model, dropout=dropout)
        self.ad_hoc_knrm_layer = KNRMLayer(d_model)
        self.feat_linear = nn.Linear(11, 1, 1)
        self.ranker = nn.Linear(7, 1, 1)
        self.bn = nn.BatchNorm1d(11)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def get_embedding(self, queries, documents):
        batch_size, n_blocks, n_tokens_q = queries.size()
        n_tokens_d = documents.size(2)

        emb = self.word_emb(queries)
        mask_query = get_pad_mask(queries, self.pad_idx).view(batch_size, n_blocks, n_tokens_q)
        mask_session = torch.sum(mask_query, -1) > 0
        word_vec_q = emb.view(batch_size, n_blocks, n_tokens_q, -1)

        # documents emb
        emb = self.word_emb(documents)
        mask_doc = get_pad_mask(documents, self.pad_idx).view(batch_size, n_blocks, n_tokens_d)
        word_vec_d = emb.view(batch_size, n_blocks, n_tokens_d, -1)

        return word_vec_q, word_vec_d, mask_query, mask_doc, mask_session

    def get_position_embedding(self, queries, documents, word_vec_q, word_vec_d):

        batch_size, n_blocks, n_tokens_q = queries.size()
        n_tokens_d = documents.size(2)

        local_pos_emb_q = self.pos_emb.pe[:, :n_tokens_q].unsqueeze(1).expand(batch_size, n_blocks, n_tokens_q,
                                                                          int(self.d_word_vec))
        emb = word_vec_q
        # emb = emb * math.sqrt(self.d_word_vec)
        emb = emb + local_pos_emb_q
        emb = self.pos_emb.dropout(emb)
        word_vec_q_encode = emb.view(batch_size * n_blocks, n_tokens_q, -1)

        local_pos_emb_d = self.pos_emb.pe[:, :n_tokens_d].unsqueeze(1).expand(batch_size, n_blocks, n_tokens_d,
                                                                          int(self.d_word_vec))
        emb = word_vec_d
        # emb = emb * math.sqrt(self.d_word_vec)
        emb = emb + local_pos_emb_d
        emb = self.pos_emb.dropout(emb)
        word_vec_d_encode = emb.view(batch_size * n_blocks, n_tokens_d, -1)
        return word_vec_q_encode, word_vec_d_encode

    def get_adjacent_seq(self, word_vec_q, word_vec_d, mask_query, mask_doc):
        qd = torch.cat([word_vec_q, word_vec_d], 2)
        qd_mask = torch.cat([mask_query, mask_doc], 2)
        prev = torch.zeros_like(qd)
        prev_mask = torch.zeros_like(qd_mask)
        next_q = torch.zeros_like(word_vec_q)
        next_q_mask = torch.zeros_like(mask_query)
        prev[:, 1:, :, :] = qd[:, :-1, :, :]
        prev_mask[:,1:,:] = qd_mask[:,:-1,:]
        next_q[:, :-1, :, :] = word_vec_q[:, 1:, :, :]
        next_q_mask[:, :-1, :] = mask_query[:, 1:, :]

        prev_session_mask = torch.sum(prev_mask, -1) > 0
        next_session_mask = torch.sum(next_q_mask, -1) > 0
        return prev, prev_mask, next_q, next_q_mask, prev_session_mask, next_session_mask

    def get_representation(self, word_vec_q_encode, mask_query, retained_atten, added_atten, removed_atten):
        retained_rep = torch.sum(word_vec_q_encode * retained_atten.unsqueeze(-1), 2)
        added_rep = torch.sum(word_vec_q_encode * added_atten.unsqueeze(-1), 2)
        removed_rep = torch.sum(word_vec_q_encode * removed_atten.unsqueeze(-1), 2)

        querylen = torch.sum(mask_query, -1).unsqueeze(-1) + 1e-4
        mean_rep = torch.sum(word_vec_q_encode, 2) / querylen

        return retained_rep, added_rep, removed_rep, mean_rep


    def forward(self, queries, documents, can_index, wss_label, features):

        batch_size, n_blocks, n_tokens_q = queries.size()
        n_tokens_d = documents.size(2)

        word_vec_q, word_vec_d, mask_query, mask_doc, mask_session = self.get_embedding(queries, documents)

        word_vec_q_encode, word_vec_d_encode = self.get_position_embedding(queries, documents, word_vec_q, word_vec_d)
        for i in range(self.num_layers):
            word_vec_q_encode = self.query_encoding_layers[i](word_vec_q_encode, mask_query)
            word_vec_d_encode = self.document_encoding_layers[i](word_vec_d_encode, mask_doc)

        word_vec_q_encode = word_vec_q_encode.view(batch_size, n_blocks, n_tokens_q, -1)
        word_vec_d_encode = word_vec_d_encode.view(batch_size, n_blocks, n_tokens_d, -1)

        prev_qd, prev_qd_mask, next_q, next_q_mask, prev_session_mask, next_session_mask = self.get_adjacent_seq(word_vec_q_encode, word_vec_d_encode, mask_query, mask_doc)

        retained_atten, added_atten, removed_atten = self.reformulation_atten(word_vec_q_encode, mask_query, prev_qd, prev_qd_mask, next_q, next_q_mask)

        retained_rep, added_rep, removed_rep, mean_rep = self.get_representation(word_vec_q, mask_query, retained_atten, added_atten, removed_atten)

        retained_rep = self.pos_emb(retained_rep)
        added_rep = self.pos_emb(added_rep)
        removed_rep = self.pos_emb(removed_rep)
        mean_rep = self.pos_emb(mean_rep)

        for i in range(self.num_layers):
            retained_rep = self.retained_encoding_layers[i](retained_rep, prev_session_mask & mask_session)
            added_rep = self.added_encoding_layers[i](added_rep, prev_session_mask & mask_session)
            removed_rep = self.removed_encoding_layers[i](removed_rep, next_session_mask & mask_session)
            mean_rep = self.mean_encoding_layers[i](mean_rep, mask_session)

        retained_rep_cur = torch.index_select(retained_rep.view(batch_size * n_blocks, -1), 0, can_index)
        added_rep_cur = torch.index_select(added_rep.view(batch_size * n_blocks, -1), 0, can_index)
        removed_rep_cur = torch.index_select(removed_rep.view(batch_size * n_blocks, -1), 0, can_index - 1)
        mean_rep_cur = torch.index_select(mean_rep.view(batch_size * n_blocks, -1), 0, can_index)
        candidate_encode = torch.index_select(word_vec_d_encode.view(batch_size * n_blocks, n_tokens_d, -1), 0, can_index)
        candidate_mask = torch.index_select(mask_doc.view(batch_size * n_blocks, n_tokens_d), 0, can_index)

        doclen = torch.sum(candidate_mask, -1).unsqueeze(-1) + 1e-4
        can_rep = torch.sum(candidate_encode, 1) / doclen
        mean_rep_score = self.rep_interaction_layer(retained_rep_cur, added_rep_cur, removed_rep_cur, mean_rep_cur, can_rep)

        current_query = torch.index_select(word_vec_q.view(batch_size * n_blocks, n_tokens_q, -1), 0, can_index)
        current_query_mask = torch.index_select(mask_query.view(batch_size * n_blocks, n_tokens_q), 0, can_index)
        candidate = torch.index_select(word_vec_d.view(batch_size * n_blocks, n_tokens_d, -1), 0, can_index)

        # ----------------------reformulation classifier -----------------------
        last_query = torch.index_select(word_vec_q.view(batch_size * n_blocks, n_tokens_q, -1), 0, can_index-1)
        last_query_mask = torch.index_select(mask_query.view(batch_size * n_blocks, n_tokens_q), 0, can_index-1)

        retained = torch.index_select(retained_atten.view(batch_size * n_blocks, n_tokens_q), 0, can_index)
        added = torch.index_select(added_atten.view(batch_size * n_blocks, n_tokens_q), 0, can_index)
        removed = torch.index_select(removed_atten.view(batch_size * n_blocks, n_tokens_q), 0, can_index-1)

        reform_logits = self.reformulation_classifier(current_query, current_query_mask, last_query, last_query_mask, retained, added, removed)
        reform_loss_fct = CrossEntropyLoss(weight=self.class_weight_reform)
        loss_reform = reform_loss_fct(reform_logits.view(-1, self.reform_type), wss_label.view(-1))
        # ----------------------reformulation classifier -----------------------

        retained_score = self.retained_attentive_knrm_layer(retained_atten, word_vec_q, candidate, retained_rep, retained_rep_cur,
                            mask_query, candidate_mask, prev_session_mask & mask_session)
        added_score = self.added_attentive_knrm_layer(added_atten, word_vec_q, candidate, added_rep, added_rep_cur,
                            mask_query, candidate_mask, prev_session_mask & mask_session)
        removed_score = self.removed_attentive_knrm_layer(removed_atten, word_vec_q, candidate, removed_rep, removed_rep_cur,
                            mask_query, candidate_mask, next_session_mask & mask_session)
        mean_score = self.mean_attentive_knrm_layer(None, word_vec_q, candidate, mean_rep, mean_rep_cur,
                            mask_query, candidate_mask, mask_session)
        ad_hoc_score = self.ad_hoc_knrm_layer(current_query, candidate, current_query_mask, candidate_mask)

        features = self.bn(features.float())
        feat_score = F.tanh(self.feat_linear(features.float())).squeeze(-1)
        scores = torch.stack((mean_rep_score, retained_score, added_score, removed_score, mean_score, ad_hoc_score, feat_score), -1)

        score = F.tanh(self.ranker(scores)).squeeze(-1)
        return score, loss_reform