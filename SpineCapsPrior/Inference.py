import argparse
from tqdm import tqdm
import os
import importlib
from pathlib import Path
import pickle
from plyfile import PlyData
import pdb

import numpy as np
import pandas as pd
from collections import defaultdict

from torch.utils.data import DataLoader
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data

from scripts.utils import load_yaml, seed_everything, init_logger
from scripts.SpineDataset import SpineDataset


def argparser():
    parser = argparse.ArgumentParser(description='SpineCaps Inference')
    parser.add_argument('--config_path', type=str, help='config_path')
    return parser.parse_args()


class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def build_checkpoints_list(cfg):
    pipeline_path = Path(cfg['CHECKPOINTS']['PIPELINE_PATH'])
    pipeline_name = cfg['CHECKPOINTS']['PIPELINE_NAME']

    checkpoints_list = []
    if cfg.get('SUBMIT_BEST', False):
        best_checkpoints_folder = Path(
            pipeline_path,
            cfg['CHECKPOINTS']['BEST_FOLDER']
        )

        usefolds = cfg['USEFOLDS']
        for fold_id in usefolds:
            filename = '{}_fold{}.pth'.format(pipeline_name, fold_id)
            checkpoints_list.append(Path(best_checkpoints_folder, filename))
    else:
        folds_dict = cfg['SELECTED_CHECKPOINTS']
        for folder_name, epoch_list in folds_dict.items():
            checkpoint_folder = Path(
                pipeline_path,
                cfg['CHECKPOINTS']['FULL_FOLDER'],
                folder_name,
            )
            for epoch in epoch_list:
                checkpoint_path = Path(
                    checkpoint_folder,
                    '{}_{}_epoch{}.pth'.format(pipeline_name, folder_name, epoch)
                )
                checkpoints_list.append(checkpoint_path)
    return checkpoints_list


def inference_image(model, data, device):
    batch = data.to(device)
    batch_priors = batch.prior.float().unsqueeze(0)
    batch_graphs = batch.pos.unsqueeze(0).transpose(2,1).contiguous()
    codewords, reconstruction = model(batch_graphs, batch_priors)

    reconstruction_ = reconstruction.transpose(2, 1).contiguous()
    return reconstruction_


def inference_model(model, loader, latent_hook, device):
    recon_list = []
    for idx, batch in enumerate(tqdm(loader)):
        recon = inference_image(model, batch['data'], device)
        batch['recon'] = recon.detach().cpu()
        batch['feature'] = latent_hook.output.detach().cpu()
        recon_list.append(batch)
    return recon_list


def main():
    pdb.set_trace()
    args = argparser()
    config_path = Path(args.config_path)
    inference_config = load_yaml(config_path)
    print(inference_config)

    SEED = inference_config['SEED']
    seed_everything(SEED)

    batch_size = inference_config['BATCH_SIZE']
    device = inference_config['DEVICE']

    module = importlib.import_module(inference_config['MODEL']['PY'])
    model_class = getattr(module, inference_config['MODEL']['CLASS'])
    model = model_class(**inference_config['MODEL'].get('ARGS', None)).to(device)
    model.eval()

    num_workers = inference_config['NUM_WORKERS']
    dataset_folder = inference_config['DATA_DIRECTORY']
    df_path = inference_config['PRIOR_CSV']
    df = pd.read_csv(df_path, index_col=0).drop(columns=['Type'])
    df = (df - df.mean()) / df.std()
    pre_transform, transform = T.Compose([T.NormalizeScale()]), T.SamplePoints(1024)
    data_list = []
    paths = [path for path in Path(dataset_folder).glob('*.ply')]
    for path in paths:
        name = path.name
        data = PlyData.read(path)
        prior = torch.tensor(df.loc[path.name].values)
        vert_data = torch.tensor(list(map(lambda x: (x[0], x[1], x[2]), data['vertex'].data)))
        face_data = torch.tensor(list(map(lambda x: (x[0].astype(np.long)), data['face'].data)))
        data = pre_transform(Data(pos=vert_data, face=face_data.T, prior=prior))
        data = transform(data)
        data_list.append({'name': name, 'data': data})

    checkpoints_list = build_checkpoints_list(inference_config)

    latent_hook = Hook(model.latent_caps_layer)
    recon_list = []
    for pred_idx, checkpoint_path in enumerate(checkpoints_list):
        print(checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        inference_recon_list = inference_model(model, data_list, latent_hook, device)
        recon_list = inference_recon_list

    result_path = Path(inference_config['RESULT_FOLDER'], inference_config['RESULT'])

    with open(result_path, 'wb') as handle:
        pickle.dump(recon_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    df = pd.DataFrame()
    for result in recon_list:
        name = result['name']
        feature = result['feature'].reshape(-1)

        df[name] = feature
    df.to_csv(result_path.parent / (result_path.stem + '.csv'), index=False)


if __name__ == "__main__":
    main()
