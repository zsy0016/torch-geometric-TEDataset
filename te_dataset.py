import os
import os.path as osp
import shutil

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url

from .io import read_te_data


class TEDataset(InMemoryDataset):
    r"""A graph dataset of the Tennessee Eastman process, collected from 
    <http://web.mit.edu/braatzgroup/links.html>. The topological structure 
    is calculated using the Convergent Cross Mapping. 
    The raw Tennessee Eastman process data can be accessed on 
    <https://github.com/zsy0016/Tennessee-Eastman-Process>.
    Mask is returned to indicate the fault type of data.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name` of the dataset.
        ebd (int): The embed dimension of TEP variables.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/zsy0016/'
           'Tennessee-Eastman-Process/master')

    def __init__(self, root, name, mode='train', ebd=10, transform=None, 
        pre_transform=None, pre_filter=None):
        self.name = name
        self.mode = mode
        self.ebd = ebd
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])
        if mode == 'train':
            self.data.x = self.data.x[self.data.m == 0]
            self.data.y = self.data.y[self.data.m == 0]
            self.data.f = self.data.f[self.data.m == 0]
            self.data.m = self.data.m[self.data.m == 0]
        else:
            self.data.x = self.data.x[self.data.m != 0]
            self.data.y = self.data.y[self.data.m != 0]
            self.data.f = self.data.f[self.data.m != 0]
            self.data.m = self.data.m[self.data.m != 0]
        for f in self.data.f.unique():
            self.data.y[torch.where(self.data.f == f)[0][:ebd]] = -1

    @property
    def raw_dir(self):
        name = 'raw'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        adj_name = 'adj'
        train_name = 'd00_te'
        test_names = [
            'd00',    'd01_te', 'd02_te', 'd03_te', 'd04_te', 'd05_te', 
            'd06_te', 'd07_te', 'd08_te', 'd09_te', 'd10_te', 'd11_te', 
            'd12_te', 'd13_te', 'd14_te', 'd15_te', 'd16_te', 'd17_te', 
            'd18_te', 'd19_te', 'd20_te', 'd21_te', 
        ]
        return adj_name, train_name, test_names

    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        adj_name, train_name, test_names = self.raw_file_names
        files = [train_name] + test_names
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def len(self):
        return (self.data.y != -1).sum().item()

    def get(self, idx):
        idx = torch.where(self.data.y != -1)[0][idx]
        edge_index = self.data.edge_index
        x = self.data.x[idx - self.ebd + 1 : idx + 1]
        x = x.transpose(1, 0)
        y = self.data.y[idx: idx+1]
        data = Data(x=x, edge_index=edge_index, y=y)
        return data
    
    def download(self):
        url = self.url
        adj_name, train_name, test_names = self.raw_file_names
        dat_names = [train_name] + test_names
        for dat_name in dat_names:
            download_url('{}/{}.dat'.format(url, dat_name), self.raw_dir)

    def process(self):
        data = read_te_data(self.raw_dir)
        torch.save(data, self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
