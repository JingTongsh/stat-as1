import os
import os.path as osp
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import Tuple


class MyDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', val_fold: int = -1):
        assert split in ['train', 'val', 'test']
        self.data_dir = data_dir
        if split == 'test':
            self.feat_files = glob.glob(osp.join(self.data_dir, '*.npy'))
        else:
            folds = glob.glob(osp.join(data_dir, 'fold*'))
            n_folds = len(folds)
            assert 1 <= val_fold <= n_folds
            val_dir = folds[val_fold - 1]
            if split == 'val':
                self.feat_files = glob.glob(osp.join(val_dir, '*.npy'.format()))
            else:
                self.feat_files = []
                for fold_dir in folds:
                    if fold_dir != val_dir:
                        self.feat_files += glob.glob(osp.join(fold_dir, '*.npy'.format()))
        print('-' * 60)
        num = len(self.feat_files)
        print(f'initialized {split} set, #files = {num}')
        assert num > 0

    def __len__(self):
        return len(self.feat_files)
        
    def __getitem__(self, idx):
        file_path = self.feat_files[idx]
        feat = np.load(file_path)
        label = int(osp.basename(file_path).split('-')[0])

        return feat, label
    
    def load_all(self) -> Tuple[np.ndarray, np.ndarray]:
        feat = []
        label = []
        for file_path in self.feat_files:
            feat.append(np.load(file_path))
            label.append(int(osp.basename(file_path).split('-')[0]))

        return np.array(feat), np.array(label)
