import os
import shutil
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from plyfile import PlyData
import warnings
warnings.filterwarnings('ignore')


class SpineDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(SpineDataset, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    def download(self):
        download_path = Path('/workspace/dataset/graph/Spine/')
        if not os.listdir(self.raw_dir):
            if os.path.exists(self.raw_dir):
                shutil.rmtree(self.raw_dir)
                shutil.copytree(download_path, self.raw_dir)
            else:
                shutil.copytree(download_path, self.raw_dir)

    def process(self):
        torch.save(self.process_set(), self.processed_paths[0])

    def process_set(self):
        data_list = []
        paths = [path for path in Path(self.raw_dir).glob('*.ply')]
        for path in paths:
            data = PlyData.read(path)
            vert_data = torch.tensor(list(map(lambda x: (x[0], x[1], x[2]), data['vertex'].data)))
            face_data = torch.tensor(list(map(lambda x: (x[0].astype(np.long)), data['face'].data)))
            #             name = path.name
            data = Data(pos=vert_data, face=face_data.T)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.prefilter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))
