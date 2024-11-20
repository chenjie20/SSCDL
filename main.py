import shutil
import numpy as np
import torch
import time
import copy
import datetime
import os
import sys
import time
import warnings
import argparse

from torch_geometric.datasets import TUDataset

from tqdm import tqdm

from models import GCNN
from utils import TUDatasetExt, loss_cl, logger, k_fold_with_validation, set_seed
from feature import FeatureExpander
from augument import mask_nodes, drop_nodes, permute_edges

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GC')

parser.add_argument('--dataset', type=str, default='MUTAG',
                    choices=['IMDB-BINARY', 'MUTAG', 'REDDIT-MULTI-5K', 'REDDIT-BINARY',
                             'PROTEINS', 'IMDB-MULTI', 'COLLAB', 'github_stargazers'], help='Dataset name')
parser.add_argument('--seed', type=int, default=10, help='Initializing random seed.')
parser.add_argument('--folds', type=int, default=10, help='Number of folds.')
parser.add_argument('--num_percents', type=int, default=3, help='percents of labeled training samples.')
parser.add_argument('--max_degree', type=int, default=100, help='Maximum degree.')
parser.add_argument('--global_pool', type=str, default='sum',
                    choices=['sum', 'mean'], help='Type of global pool.')
parser.add_argument('--encoder_type', type=str, default='ResGCN',
                    choices=['ResGCN', 'GCN'], help='Backbone network.')
parser.add_argument("--temperature_l", type=float, default=0.2)

parser.add_argument('--aug_type', type=str, default='nodeDropped',
                    choices=['nodeDropped', 'nodeMasked'], help='The type of augmentation.')
parser.add_argument('--aug_ratio', type=float, default=0.2, choices=[0.1, 0.2, 0.3, 0.35])
parser.add_argument('--ratio_times', type=int, default=2)

parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initializing learning rate chosen from [1e-3, 5e-3, 1e-2, 5e-2]')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Initializing weight decay.')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_feats', type=int, default=128)
parser.add_argument('--num_layers_conv', type=int, default=2)
parser.add_argument('--num_layers_fc', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.5, help='Regularization parameter.')
parser.add_argument('--beta', type=float, default=0.05, help='Regularization parameter.')

parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--residual', type=bool, default=False)

parser.add_argument('--gpu', default=0, type=int, help='GPU device idx.')

args = parser.parse_args()

torch.cuda.set_device(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def test(model, test_data):
    model.eval()

    graph_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    correct = 0
    with torch.no_grad():
        for batch_data in graph_data_loader:
            out = model.test(batch_data)
            pred = out.max(1)[1]
            correct += pred.eq(batch_data.y.view(-1)).sum().item()

    acc = correct / len(test_data)

    return acc


def batch_loader(train_data):
    train_data_ori = train_data.shuffle()
    train_data_w = copy.deepcopy(train_data_ori)
    train_data_w = mask_nodes(train_data_w, args.aug_ratio)
    # train_data_w = permute_edges(train_data_w, args.aug_ratio)

    train_data_s = copy.deepcopy(train_data_ori)
    train_data_s = mask_nodes(train_data_s, args.aug_ratio * args.ratio_times)
    # train_data_s = permute_edges(train_data_s, args.aug_ratio * args.ratio_times)

    loader_ori = DataLoader(train_data_ori, batch_size=args.batch_size, shuffle=False)
    loader_w = DataLoader(train_data_w, batch_size=args.batch_size, shuffle=False)
    loader_s = DataLoader(train_data_s, batch_size=args.batch_size, shuffle=False)
    return loader_ori, loader_w, loader_s


def pre_training(model, optimizer, train_data):
    model.train()

    loader_ori, loader_w, loader_s = batch_loader(train_data)

    total_loss = 0
    for batch_idx, (batch_ori, batch_w) in enumerate(zip(loader_ori, loader_w)):
        _, _, proj1, proj2 = model.forward_features(batch_ori, batch_w)
        loss = loss_cl(proj1, proj2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_ori.num_graphs

    return total_loss / len(train_data)


def fine_tuning(model, optimizer, train_data):
    model.train()

    loader_ori, loader_w, loader_s = batch_loader(train_data)

    total_loss = 0

    for batch_idx, (batch_ori, batch_w, batch_s) in enumerate(zip(loader_ori, loader_w, loader_s)):
        pred, proj1, proj2, logits_1, logits_2 = model(batch_ori, batch_w, batch_s)

        loss_1 = F.nll_loss(pred, batch_ori.y)
        loss_2 = loss_cl(proj1, proj2)
        loss_3 = -torch.mean(torch.sum(torch.log(logits_1) * logits_2, dim=1))
        loss = loss_1 + args.alpha * loss_2 + args.beta * loss_3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_ori.num_graphs

    return total_loss / len(train_data)


def k_fold(train_unlabel_idx_set, train_label_idx_set, val_idx_set, test_idx_set, folds=10):
    val_accs, test_accs, durations = [], [], []

    interval_size = 10
    tqdm_size = int(args.epochs / interval_size)

    for fold_idx, (train_unlabel_idx, train_label_idx, test_idx, val_idx) in (
            enumerate(zip(train_unlabel_idx_set, train_label_idx_set, test_idx_set, val_idx_set))):

        train_unlabel_idx[train_unlabel_idx < 0] = train_unlabel_idx[0]
        train_unlabel_idx[train_unlabel_idx >= len(dataset)] = train_unlabel_idx[0]
        train_label_idx[train_label_idx < 0] = train_label_idx[0]
        train_label_idx[train_label_idx >= len(dataset)] = train_label_idx[0]
        test_idx[test_idx < 0] = test_idx[0]
        test_idx[test_idx >= len(dataset)] = test_idx[0]
        val_idx[val_idx < 0] = val_idx[0]
        val_idx[val_idx >= len(dataset)] = val_idx[0]

        train_unlabel_dataset = dataset[train_unlabel_idx]
        train_label_dataset = dataset[train_label_idx]
        test_dataset = dataset[test_idx.to(dtype=torch.int64)]
        val_dataset = dataset[val_idx.to(dtype=torch.int64)]

        model = GCNN(num_node_features,
                     args.hidden_feats,
                     num_classes,
                     args.encoder_type,
                     args.num_layers_conv,
                     args.num_layers_fc,
                     args.global_pool,
                     args.dropout,
                     args.temperature_l,
                     args.residual).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        t_start = time.perf_counter()

        for epoch in range(args.epochs):
            train_loss = pre_training(model, optimizer, train_unlabel_dataset)
            if epoch % 50 == 0:
                print('Pre training, epoch {} loss:{:.7f}'.format(epoch, train_loss))

        # pbar = tqdm(range(1, args.epochs+1), desc=f'Fine tuning on fold {fold_idx}',
        #             bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', ncols=120)
        test_acc = 0
        val_acc = 0
        # for epoch in pbar:
        for epoch in range(1, args.epochs + 1):
            train_loss = fine_tuning(model, optimizer, train_label_dataset)
            if epoch % interval_size == 0:
                test_acc = test(model, test_dataset)
                test_accs.append(test_acc)
                val_acc = test(model, val_dataset)
                val_accs.append(val_acc)
                model_file_path_tmp = model_dir_tmp + '/' + str(fold_idx)
                if not os.path.exists(model_file_path_tmp):
                    os.mkdir(model_file_path_tmp)
                torch.save(model.state_dict(), model_file_path_tmp + '/' + str(epoch) + '_model.pth')
                print('Fine-tuning, epoch {} loss:{:.7f} val_acc:{:.2f} test_acc:{:.2f}'.format(epoch, train_loss,
                                                                                                val_acc, test_acc))
            # pbar.set_postfix(loss=f'{train_loss:.3f}', val_acc=f'{val_acc:.2f}', test_acc=f'{test_acc:.2f}')

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = torch.tensor(durations)
    test_acc_reuslts = torch.tensor(test_accs)
    val_acc_reuslts = torch.tensor(val_accs)
    test_acc_reuslts = test_acc_reuslts.view(folds, tqdm_size)
    val_acc_reuslts = val_acc_reuslts.view(folds, tqdm_size)
    _, selected_epoches = val_acc_reuslts.max(dim=1)

    test_acc_list = test_acc_reuslts[torch.arange(folds, dtype=torch.long), selected_epoches]

    test_acc_mean = test_acc_list.mean().item()
    test_acc_std = test_acc_list.std().item()

    test_acc_mean_v = test_acc_reuslts[torch.arange(folds, dtype=torch.long), -1].mean().item()
    test_acc_std_v = test_acc_reuslts[torch.arange(folds, dtype=torch.long), -1].std().item()

    duration = duration.sum().item()

    for fold_idx in range(folds):
        model_file_path_tmp = model_dir_tmp + '/' + str(fold_idx) + '/' + str(
            (selected_epoches[fold_idx].item() + 1) * interval_size) + '_model.pth'
        model_name = model_name_format % (args.dataset, args.num_percents, fold_idx)
        os.rename(model_file_path_tmp, model_dir_tmp + '/' + str(fold_idx) + '/' + model_name)
        shutil.move(model_dir_tmp + '/' + str(fold_idx) + '/' + model_name, model_dir + '/' + model_name)
    shutil.rmtree(model_dir_tmp)

    print("Final testing result, acc:{:.4f}, std:{:.4f} | without validation acc:{:.4f}, std:{:.4f}".format(
        test_acc_mean, test_acc_std, test_acc_mean_v, test_acc_std_v))

    return test_acc_mean, test_acc_std, test_acc_mean_v, test_acc_std_v, duration


def set_degree():
    if args.dataset in ['REDDIT-BINARY', 'REDDIT-MULTI-5K', 'github_stargazers']:
        args.max_degree = 10
    else:
        args.max_degree = 100


def create_dir_model(model_parent_dir, model_parent_dir_tmp):
    record_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
    model_dir = model_parent_dir + record_time
    model_dir_tmp = model_parent_dir_tmp + record_time
    if not os.path.exists('./models'):
        os.mkdir('./models')
    if not os.path.exists(model_parent_dir):
        os.mkdir(model_parent_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(model_parent_dir_tmp):
        os.mkdir(model_parent_dir_tmp)
    if not os.path.exists(model_dir_tmp):
        os.mkdir(model_dir_tmp)

    return model_dir, model_dir_tmp, record_time


if __name__ == '__main__':

    set_seed(args.seed)

    set_degree()
    pre_transform = FeatureExpander(
        degree=True, onehot_maxdeg=args.max_degree, AK=0,
        centrality=False, remove_edges='none',
        edge_noises_add=0, edge_noises_delete=0,
        group_degree=0).transform

    dataset = TUDataset(
        './data', args.dataset, pre_transform=pre_transform,
        use_node_attr=False)  #True False

    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes
    model_name_format = 'GC_pytorch_model_%s_%d_%d.pth'
    model_parent_dir = './models/models_%s_%d/' % (args.dataset, args.gpu)
    model_parent_dir_tmp = './models/models_tmp_%s_%d/' % (args.dataset, args.gpu)

    percents = np.array([3, 5, 7], dtype=np.int32)
    learning_rates = np.array([5e-3], dtype=np.float32)  #5e-3, 1e-4, 5e-4, 1e-2
    dims_layers = np.array([128, 64], dtype=np.int32)
    epochs_array = np.array([200], dtype=np.int32)
    batch_sizes = np.array([128, 64], dtype=np.int32)

    conv_num_layers = np.array([2, 3], dtype=np.int32)
    mlp_num_layers = np.array([1, 2, 3], dtype=np.int32)
    aug_ratios = np.array([0.2], dtype=np.float32)  #0.1, 0.2, 0.3, 0.35
    alphas = np.array([0.05, 0.1, 0.5, 1], dtype=np.float32)
    betas = np.array([0.05, 0.1, 0.5, 1], dtype=np.float32)

    model_dir, model_dir_tmp, record_time = create_dir_model(model_parent_dir,
                                                             model_parent_dir_tmp)

    enable_param_combination = True
    if not enable_param_combination:
        _, train_unlabel_idx_set, train_label_idx_set, val_idx_set, test_idx_set \
            = k_fold_with_validation(dataset, dataset.y, args.folds, args.num_percents)
        dataset.data.to(device)
        (results_mean, results_std,
         test_acc_mean_v, test_acc_std_v, _) = k_fold(train_unlabel_idx_set,
                                                      train_label_idx_set,
                                                      val_idx_set,
                                                      test_idx_set,
                                                      args.folds)
        with open('%s_result_%s.txt' % (args.dataset, args.gpu), 'a+') as f:
            f.write('\t {} {:.4f} \t {:.2f} \t {} \t {} \t {} \t {:.4f} '
                    ' \t {:.4f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \n'.format(
                args.num_percents, args.lr, args.aug_ratio, args.epochs, args.hidden_feats,
                args.batch_size, args.alpha, args.beta, args.num_layers_conv,
                args.num_layers_fc, record_time, results_mean, results_std, test_acc_mean_v,
                test_acc_std_v))
            f.flush()
    else:
        for percent_idx in range(percents.shape[0]):
            args.num_percents = percents[percent_idx]
            _, train_unlabel_idx_set, train_label_idx_set, val_idx_set, test_idx_set \
                = k_fold_with_validation(dataset, dataset.y, args.folds, args.num_percents)
            dataset.data.to(device)

            for lr_idx in range(learning_rates.shape[0]):
                args.lr = learning_rates[lr_idx]
                for rt_idx in range(aug_ratios.shape[0]):
                    args.aug_ratio = aug_ratios[rt_idx]
                    for dim_idx in range(dims_layers.shape[0]):
                        args.hidden_feats = dims_layers[dim_idx]
                        for epoch_idx in range(epochs_array.shape[0]):
                            args.epochs = epochs_array[epoch_idx]
                            for batch_idx in range(batch_sizes.shape[0]):
                                args.batch_size = int(batch_sizes[batch_idx])
                                for conv_idx in range(conv_num_layers.shape[0]):
                                    args.num_layers_conv = int(conv_num_layers[conv_idx])
                                    for fc_idx in range(mlp_num_layers.shape[0]):
                                        args.num_layers_fc = mlp_num_layers[fc_idx]
                                        for alpha_idx in range(alphas.shape[0]):
                                            args.alpha = alphas[alpha_idx]
                                            for beta_idx in range(betas.shape[0]):
                                                args.beta = betas[beta_idx]
                                                model_dir, model_dir_tmp, record_time = create_dir_model(
                                                    model_parent_dir,
                                                    model_parent_dir_tmp)
                                                (results_mean, results_std,
                                                 test_acc_mean_v, test_acc_std_v, _) = k_fold(train_unlabel_idx_set,
                                                                                              train_label_idx_set,
                                                                                              val_idx_set,
                                                                                              test_idx_set,
                                                                                              args.folds)
                                                with open('%s_result_%s.txt' % (args.dataset, args.gpu), 'a+') as f:
                                                    f.write('\t {} {:.4f} \t {:.2f} \t {} \t {} \t {} \t {:.4f} '
                                                            ' \t {:.4f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \n'.format(
                                                        args.num_percents, args.lr, args.aug_ratio, args.epochs,
                                                        args.hidden_feats,
                                                        args.batch_size, args.alpha, args.beta, args.num_layers_conv,
                                                        args.num_layers_fc, record_time, results_mean, results_std,
                                                        test_acc_mean_v,
                                                        test_acc_std_v))
                                                    f.flush()
