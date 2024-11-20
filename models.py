import torch.nn as nn
from layers import GCN, ResGCN
import torch.nn.functional as F
import torch
from torch.nn import Linear

from sklearn.metrics.pairwise import cosine_similarity

class GCNN(nn.Module):

    def __init__(self, in_feats, h_feats, out_feats, encoder_type, num_conv_layers,
                 num_fc_layers, global_pool, dropout=0, temperature_l=1.0, residual=False):
        super(GCNN, self).__init__()
        self.dropout = dropout
        self.temperature_l = temperature_l
        if encoder_type == 'GCN':
            self.encoder = GCN(in_feats, h_feats, h_feats, global_pool)
        else:
            self.encoder = ResGCN(in_feats, h_feats, global_pool, num_conv_layers, num_fc_layers, residual)

        self.lin_class = Linear(h_feats, out_feats)

        # projection
        self.proj_head = nn.Sequential(nn.Linear(h_feats, h_feats),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(h_feats, h_feats))

    def forward_features(self, data, data_w):

        h_ori = self.encoder(data)
        proj1 = self.projector(h_ori)
        h_w = self.encoder(data_w)
        proj2 = self.projector(h_w)

        return h_ori, h_w, proj1, proj2

    def forward(self, data, data_w, data_s):

        h_ori, h_w, proj1, proj2 = self.forward_features(data, data_w)

        pred = self.predictor(h_ori)

        # normalization for original features
        h_ori_n = nn.functional.normalize(h_ori, dim=1)

        # weak features  p(h_w|h)
        h_w_n = nn.functional.normalize(h_w, dim=1)
        l_1 = torch.einsum('nc,ck->nk', [h_w_n, h_ori_n.T])
        logits_1 = l_1 / self.temperature_l
        logits_1 = torch.softmax(logits_1, dim=1)

        # strong features p(h_s|h)
        h_s = self.encoder(data_s)
        h_s_n = nn.functional.normalize(h_s, dim=1)
        l_2 = torch.einsum('nc,ck->nk', [h_s_n, h_ori_n.T])
        logits_2 = l_2 / self.temperature_l
        logits_2 = torch.softmax(logits_2, dim=1)

        return pred, proj1, proj2, logits_1, logits_2

    # def forward(self, data, data_w, data_s):
    #
    #     h_ori, h_w, proj1, proj2 = self.forward_features(data, data_w)
    #
    #     pred = self.predictor(h_ori)
    #
    #     # normalization for original and weak features
    #     h_ori_n = h_ori.norm(dim=1)
    #     h_w_n = h_w.norm(dim=1)
    #
    #     # strong features
    #     h_s = self.encoder_k(data_s)
    #     h_s_n = h_s.norm(dim=1)
    #
    #     # p(h_w|h)
    #     q_1 = torch.einsum('nc,ck->nk', [h_w, h_ori.T])
    #     q_1_n = torch.einsum('i,j->ij', [h_w_n, h_ori_n.T])
    #     logits_1 = q_1 / q_1_n
    #     logits_1 = logits_1 / self.temperature_l
    #     logits_1 = torch.softmax(logits_1, dim=1)
    #
    #     # p(h_s|h)
    #     q_2 = torch.einsum('nc,ck->nk', [h_s, h_ori.T])
    #     q_2_n = torch.einsum('i,j->ij', [h_s_n, h_ori_n.T])
    #     logits_2 = q_2 / q_2_n
    #     logits_2 = logits_2 / self.temperature_l
    #     logits_2 = torch.softmax(logits_2, dim=1)
    #
    #     return pred, proj1, proj2, logits_1, logits_2

    def test(self, data):
        graph_k = self.encoder(data)
        pred = self.predictor(graph_k)
        return pred

    def predictor(self, x):
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.lin_class(x)
        pred = F.log_softmax(out, dim=-1)
        return pred

    def projector(self, x):
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        proj = self.proj_head(x)
        return proj

