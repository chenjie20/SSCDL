import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool


class GCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, global_pool):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, out_feats)
        if global_pool == 'sum':
            self.pool = global_add_pool
        else:
            self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        g = self.pool(x, data.batch)
        return g


# class ResGCN(nn.Module):
#
#     def __init__(self, in_feats, h_feats, global_pool, num_conv_layers, num_fc_layers, residual=False):
#         super(ResGCN, self).__init__()
#         self.activation = nn.ReLU()
#         self.conv_residual = residual
#
#         self.layers = torch.nn.ModuleList()
#         self.layers.append(nn.BatchNorm1d(in_feats))
#         self.layers.append(Linear(in_feats, h_feats))
#         for i in range(num_conv_layers):
#             self.layers.append(nn.BatchNorm1d(h_feats))
#             self.layers.append(GCNConv(h_feats, h_feats))
#
#         if global_pool == 'sum':
#             self.global_pool = global_add_pool
#         else:
#             self.global_pool = global_mean_pool
#
#         self.lins = torch.nn.ModuleList()
#         for i in range(num_fc_layers):
#             self.lins.append(nn.BatchNorm1d(h_feats))
#             self.lins.append(nn.Linear(h_feats, h_feats))
#
#         self.norm_layer = nn.BatchNorm1d(h_feats)
#
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         for i, layer in enumerate(self.layers):
#             if i % 2 == 0:
#                 x = layer(x)
#             else:
#                 if i == 1:
#                     x = layer(x)
#                 else:
#                     x_n = layer(x, edge_index)
#                     x = x + x_n if self.conv_residual else x_n
#                 x = self.activation(x)
#
#         # graph representation
#         x = self.global_pool(x, batch)
#
#         for i, layer in enumerate(self.lins):
#             if i % 2 == 0:
#                 x = layer(x)
#             else:
#                 x = layer(x)
#                 x = self.activation(x)
#
#         x = self.norm_layer(x)
#
#         return x

class ResGCN(nn.Module):

    def __init__(self, in_feats, h_feats, global_pool, num_conv_layers, num_fc_layers, residual=False):
        super(ResGCN, self).__init__()
        self.conv_residual = residual
        self.bn_feat = BatchNorm1d(in_feats)
        self.lin_feat = Linear(in_feats, h_feats)

        self.convs = nn.ModuleList()
        self.bns_conv = nn.ModuleList()
        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(h_feats))
            self.convs.append(GCNConv(h_feats, h_feats))
        if global_pool == "sum":
            self.pool = global_add_pool
        else:
            self.pool = global_mean_pool

        self.bn_hidden = BatchNorm1d(h_feats)

        self.lins = nn.ModuleList()
        self.bns_fc = torch.nn.ModuleList()
        for i in range(num_fc_layers):
            self.bns_fc.append(BatchNorm1d(h_feats))
            self.lins.append(Linear(h_feats, h_feats))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.bn_feat(x)
        x = F.relu(self.lin_feat(x))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        x = self.pool(x, batch)

        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x = F.relu(lin(x_))

        x = self.bn_hidden(x)

        return x
