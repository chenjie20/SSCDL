import torch
import numpy as np
from itertools import repeat
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import torch_geometric.utils as tg_utils


class Auguments(BaseTransform):
    def __init__(self, aug_type, aug_ratio):
        self.aug_type = aug_type
        self.aug_ratio = aug_ratio

    def __call__(self, graph: Data) -> Data:
        # l = ['dropN', 'wdropN', 'permE', 'subgraph', 'maskN', 'none']
        # self.aug_type = l[0]
        # self.aug_ratio = 0.2

        if self.aug_type == 'nodeDropped':
            graph = drop_nodes(graph, self.aug_ratio)
        elif self.aug_type == 'nodeMasked':
            graph = mask_nodes(graph, self.aug_ratio)
        elif self.aug_type == 'none':
            graph = graph
        else:
            print('augmentation error')
            assert False

        return graph


def permute_edges(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index.detach().cpu().numpy()

    idx_add = np.random.choice(node_num, (2, permute_num))

    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]
    # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add

    edge_index = np.concatenate((edge_index[:, np.random.choice(edge_num, (edge_num - permute_num), replace=False)], idx_add), axis=1)
    data.edge_index = torch.tensor(edge_index)

    return data

def drop_nodes(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.cpu().numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    try:
        # data.batch = data.batch[idx_nondrop]
        data.data.edge_index = edge_index
        data.data.x = data.x[idx_nondrop]
        data.data.num_nodes = data.x.size(0)
    except:
        data = data
    # return data.edge_index, data.x, data.num_nodes
    return data


def mask_nodes(data, aug_ratio):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.data.x[idx_mask] = token.clone().detach().to(torch.float32)

    return data