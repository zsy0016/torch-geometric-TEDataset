import glob
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce

adj_name = 'adj'

train_name = 'd00_te'

test_names = [
    'd00',    'd01_te', 'd02_te', 'd03_te', 'd04_te', 'd05_te', 
    'd06_te', 'd07_te', 'd08_te', 'd09_te', 'd10_te', 'd11_te', 
    'd12_te', 'd13_te', 'd14_te', 'd15_te', 'd16_te', 'd17_te', 
    'd18_te', 'd19_te', 'd20_te', 'd21_te', 
]

fault_origin = 160

def read_te_data(folder):
    adj = read_file(folder, adj_name)
    adj += torch.eye(adj.shape[0], dtype=torch.long)
    edge_index = torch.cat([t[None] for t in torch.where(adj == 1)])

    train_x = read_file(folder, train_name)
    test_x = [read_file(folder, test_name) for test_name in test_names]

    mean = train_x.mean(dim=0, keepdims=True)
    std = train_x.std(dim=0, keepdims=True)
    train_x = (train_x - mean) / std
    train_y = torch.zeros(len(train_x), dtype=torch.long)
    train_f = torch.zeros(len(train_x), dtype=torch.long)
    train_m = torch.zeros(len(train_x), dtype=torch.long)

    test_x = [(test_x[i] - mean) / std for i in range(len(test_x))]
    test_y = [torch.ones(len(test_x[i]), dtype=torch.long) * i 
        for i in range(len(test_x))]
    for i in range(len(test_y)):
        test_y[i][:fault_origin] = 0
    test_f = [torch.ones(len(test_x[i]), dtype=torch.long) * i 
        for i in range(len(test_x))]
    test_m = [torch.ones(len(test_x[i]), dtype=torch.long) 
        for i in range(len(test_x))]
    
    x = torch.cat([train_x] + test_x, dim=0)
    y = torch.cat([train_y] + test_y, dim=0)
    f = torch.cat([train_f] + test_f, dim=0)
    m = torch.cat([train_m] + test_m, dim=0)

    return Data(x=x, edge_index=edge_index, y=y, m=m, f=f)


def read_file(folder, name, dtype=None):
    path = osp.join(folder, '{}.dat'.format(name))
    return read_txt_array(path, sep=None, dtype=dtype)
