import copy
import random
import sys
import os

from torch_geometric.data.separate import separate
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np
from itertools import repeat
from augument import drop_nodes, mask_nodes
import sys


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def k_fold_with_validation(data, labels, folds=10, num_percents=1, random_st=100):

    num_splits = folds - 2
    assert num_splits > num_percents >= 1

    skf = StratifiedKFold(folds, shuffle=True, random_state=random_st)
    test_idx_set = []
    for _, test_indices in skf.split(data, labels):
        test_idx_set.append(torch.from_numpy(test_indices))

    val_idx_set = [test_idx_set[i - 1] for i in range(folds)]

    train_idx_set, train_unlabel_idx_set, train_label_idx_set = [], [], []

    skf_semi = StratifiedKFold(num_splits, shuffle=True, random_state=random_st)
    for i in range(folds):
        train_mask = torch.ones(len(data), dtype=torch.uint8)
        train_mask[test_idx_set[i].long()] = 0
        train_mask[val_idx_set[i].long()] = 0
        train_all_idx_set = train_mask.nonzero(as_tuple=False).view(-1)
        train_idx_set.append(train_all_idx_set)

        train_unlabel_idx_lst, train_label_idx_lst = [], []
        for _, train_label_indices \
                in skf_semi.split(torch.zeros(train_all_idx_set.size()[0]), labels[train_all_idx_set]):
            train_label_idx_lst.append(train_all_idx_set[train_label_indices])
            if len(train_label_idx_lst) >= num_percents:
                break

        train_label_idx_item = torch.concat(train_label_idx_lst).view(-1)
        train_label_idx_set.append(train_label_idx_item)

        train_mask = torch.ones(len(data), dtype=torch.uint8)
        train_mask[train_label_idx_set[i].long()] = 0
        train_unlabel_idx_item = train_mask.nonzero(as_tuple=False).view(-1)
        train_unlabel_idx_set.append(train_unlabel_idx_item)

    return train_idx_set, train_unlabel_idx_set, train_label_idx_set, val_idx_set, test_idx_set



class TUDatasetExt(TUDataset):
    def __init__(self,
                 root,
                 name,
                 degree,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr=False,
                 aug_type="none", aug_ratio=None):
        self.degree = str(degree)
        self.aug_type = aug_type
        self.aug_ratio = aug_ratio
        super(TUDatasetExt, self).__init__(root, name, transform, pre_transform,
                                           pre_filter, use_node_attr)

    @property
    def processed_file_names(self):
        return 'data_deg' + self.degree + '.pt'

    def get(self, idx):

        data = self._data.__class__()

        # 若数据没有预处理，则先读取raw数据
        if data is None or not hasattr(data,"x") or data.x is None:
            # TODO (matthias) Avoid unnecessary copy here.
            if self.len() == 1:
                return copy.copy(self._data)

            if not hasattr(self, '_data_list') or self._data_list is None:
                self._data_list = self.len() * [None]
            elif self._data_list[idx] is not None:
                return copy.copy(self._data_list[idx])

            data = separate(
                cls=self._data.__class__,
                batch=self._data,
                idx=idx,
                slice_dict=self.slices,
                decrement=False,
            )

            self._data_list[idx] = copy.copy(data)

            # return data

        # b = self.data.keys()
        for key in self._data.keys():
            if key == 'num_nodes':
                continue
            item, slices = self._data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self._data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])

            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        data['num_nodes'] = data.x.size(0)

        if self.aug_type == 'nodeDropped':
            graph = drop_nodes(data, self.aug_ratio)
        elif self.aug_type == 'nodeMasked':
            graph = mask_nodes(data, self.aug_ratio)
        elif self.aug_type == 'none':
            graph = data
        else:
            print('augmentation error')
            assert False

        return graph


def data_split(len, label_ratio):
    arr = np.arange(len)
    np.random.shuffle(arr)
    idx = torch.as_tensor(arr, dtype=torch.int64)
    train_num = int(label_ratio * len)
    test_num = int(0.1 * len)
    train_idx = idx[:train_num]
    test_idx = idx[len - test_num:]
    val_idx = test_idx
    return train_idx, test_idx, val_idx


def k_fold(dataset, folds, epoch_select, dataset_name, n_percents=1):
    n_splits = folds - 2

    if n_percents == 10:
        all_indices = torch.arange(0, len(dataset), 1, dtype=torch.long)
        return [all_indices], [all_indices], [all_indices], [all_indices]

    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices, train_indices_unlabel = [], [], []
    save_test, save_train, save_val, save_train_unlabel = [], [], [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset._data.y):
        test_indices.append(torch.from_numpy(idx))
        if len(save_test) > 0 and len(list(idx)) < len(save_test[0]):
            save_test.append(list(idx) + [list(idx)[-1]])
        else:
            save_test.append(list(idx))

    val_indices = [test_indices[i] for i in range(folds)]
    save_val = [save_test[i] for i in range(folds)]
    n_splits += 1

    skf_semi = StratifiedKFold(n_splits, shuffle=True, random_state=12345)
    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        idx_train_all = train_mask.nonzero(as_tuple=False).view(-1)

        idx_train = []
        for _, idx in skf_semi.split(torch.zeros(idx_train_all.size()[0]), dataset._data.y[idx_train_all]):
            idx_train.append(idx_train_all[idx])
            if len(idx_train) >= n_percents:
                break
        idx_train = torch.concat(idx_train).view(-1)

        train_indices.append(idx_train)
        cur_idx = list(idx_train.cpu().detach().numpy())
        if i > 0 and len(cur_idx) < len(save_train[0]):
            save_train.append(cur_idx + [cur_idx[-1]])
        else:
            save_train.append(cur_idx)

        # train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[train_indices[i].long()] = 0
        idx_train_unlabel = train_mask.nonzero(as_tuple=False).view(-1)
        train_indices_unlabel.append(idx_train_unlabel)  # idx_train_all, idx_train_unlabel
        cur_idx = list(idx_train_unlabel.cpu().detach().numpy())
        if i > 0 and len(cur_idx) < len(save_train_unlabel[0]):
            save_train_unlabel.append(cur_idx + [cur_idx[-1]])
        else:
            save_train_unlabel.append(cur_idx)

    print("Train:", len(train_indices[i]), "Val:", len(val_indices[i]), "Test:", len(test_indices[i]))

    return train_indices, test_indices, val_indices, train_indices_unlabel


def logger(info):
    fold, epoch = info['fold'], info['epoch']
    if epoch == 1 or epoch % 10 == 0:
        train_acc, test_acc = info['train_acc'], info['test_acc']
        print('{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f}'.format(
            fold, epoch, train_acc, test_acc))
    sys.stdout.flush()

# contrastive learning
def loss_cl(x1, x2):
    T = 0.5
    batch_size, _ = x1.size()

    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss1).mean()

    return loss





