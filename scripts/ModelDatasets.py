import os
import glob
import math
from collections import deque
from tqdm import tqdm_notebook
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.transforms import LinearTransformation
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.io import read_off
from scripts.transforms import Rotate


class RotModelNet(ModelNet):
    def __init__(self, root, name='10', train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        super().__init__(root, name, train, transform, pre_transform, pre_filter)
        self.rot_list = [(0, 0.), (0, 90.), (0, 180.), (0, 270.),
                         (1, 0.), (1, 90.), (1, 180.), (1, 270.),
                         (2, 0.), (2, 90.), (2, 180.), (2, 270.), ]

    '''
    def process_set(self, dataset):
        categories = glob.glob(os.path.join(self.raw_dir, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])

        data_list = deque()
        for target, category in tqdm_notebook(enumerate(categories)):
            folder = os.path.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/{}_*.off'.format(folder, category))
            for path in paths:
                rot_list = [(0, 0.), (0, 90.), (0, 180.), (0, 270.),
                            (1, 0.), (1, 90.), (1, 180.), (1, 270.),
                            (2, 0.), (2, 90.), (2, 180.), (2, 270.),]
                for rot_idx, rot in enumerate(rot_list):
                    data = read_off(path)
                    data = Rotate(rot[1], axis=rot[0])
                    data.y = torch.tensor([rot_idx])
                    data.category = torch.tensor([target])
                    data_list.append(data)
            print(category)
        data_list = list(data_list)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)
    '''

    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given).
        Returns a data object, if :obj:`idx` is a scalar, and a new dataset in
        case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a LongTensor
        or a BoolTensor."""
        if isinstance(idx, int):
            data = self.get(idx)
            data = data if self.transform is None else self.transform(data)

            rot_idx = np.random.randint(0, len(self.rot_list))
            data = Rotate(self.rot_list[rot_idx][1], self.rot_list[rot_idx][0])(data)
            data.category = data.y
            data.y = rot_idx
            return data
        elif isinstance(idx, slice):
            return self.__indexing__(range(*idx.indices(len(self))))
        elif torch.is_tensor(idx):
            if idx.dtype == torch.long:
                return self.__indexing__(idx)
            elif idx.dtype == torch.bool or idx.dtype == torch.uint8:
                return self.__indexing__(idx.nonzero())

        raise IndexError(
            'Only integers, slices (`:`) and long or bool tensors are valid '
            'indices (got {}).'.format(type(idx).__name__))