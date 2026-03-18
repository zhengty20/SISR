import os
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def default_patch_size(scale):
    return {2: 144, 3: 192, 4: 192}[scale]


def collect_shard_files(root_dir):
    return sorted(str(path) for path in Path(root_dir).rglob('*.pt'))

class SRTrainDataset(Dataset):
    def __init__(self, train_dir, scale, patch_size=0):
        self.scale = scale
        self.patch_size = patch_size if patch_size > 0 else default_patch_size(scale)
        if self.patch_size % scale != 0:
            raise ValueError(f'patch_size={self.patch_size} 必须能被 scale={scale} 整除')

        train_files = collect_shard_files(train_dir)
        if not train_files:
            raise ValueError(f'未在 {train_dir} 找到新的训练分片，请先重新运行 image_processor_train.py')

        self.hr_data = []
        self.lr_data = []
        self.dataset_names = []

        for file_path in train_files:
            packed = torch.load(file_path, weights_only=False)
            lr_key = f'lr_x{scale}'
            if not isinstance(packed, dict) or 'hr' not in packed or lr_key not in packed or 'names' not in packed or 'dataset_names' not in packed:
                raise ValueError(f'{file_path} 不是新的训练分片格式')

            hr_batch = packed['hr']
            lr_batch = packed[lr_key]
            names = packed['names']
            dataset_names = packed['dataset_names']

            if len(hr_batch) != len(lr_batch) or len(hr_batch) != len(names) or len(hr_batch) != len(dataset_names):
                raise ValueError(f'{file_path} 中 names/dataset_names/HR/LR 数量不一致')

            self.hr_data.extend(list(hr_batch))
            self.lr_data.extend(list(lr_batch))
            self.dataset_names.extend(list(dataset_names))

        dataset_counts = Counter(self.dataset_names)
        self.sample_weights = torch.tensor(
            [1.0 / dataset_counts[name] for name in self.dataset_names],
            dtype=torch.double
        )
    
    def __len__(self):
        return len(self.hr_data)

    def _paired_random_crop(self, lr, hr):
        lr_patch = self.patch_size // self.scale
        lr_h, lr_w = lr.shape[-2:]

        if lr_h < lr_patch or lr_w < lr_patch:
            raise ValueError(f'LR patch 尺寸过小: {(lr_h, lr_w)} < {lr_patch}')

        if lr_h == lr_patch and lr_w == lr_patch:
            top = 0
            left = 0
        else:
            top = random.randint(0, lr_h - lr_patch)
            left = random.randint(0, lr_w - lr_patch)

        hr_top = top * self.scale
        hr_left = left * self.scale
        hr_patch = self.patch_size

        lr = lr[:, top:top + lr_patch, left:left + lr_patch]
        hr = hr[:, hr_top:hr_top + hr_patch, hr_left:hr_left + hr_patch]
        return lr, hr

    def _paired_augment(self, lr, hr):
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[2])
            hr = torch.flip(hr, dims=[2])

        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[1])
            hr = torch.flip(hr, dims=[1])

        k = random.randint(0, 3)
        if k:
            lr = torch.rot90(lr, k, dims=[1, 2])
            hr = torch.rot90(hr, k, dims=[1, 2])

        return lr.contiguous(), hr.contiguous()
    
    def __getitem__(self, idx):
        lr = self.lr_data[idx].clone()
        hr = self.hr_data[idx].clone()
        lr, hr = self._paired_random_crop(lr, hr)
        lr, hr = self._paired_augment(lr, hr)
        return lr, hr

class SRValDataset(Dataset):
    def __init__(self, val_dir, scale):
        shard_files = sorted(
            os.path.join(val_dir, file_name)
            for file_name in os.listdir(val_dir)
            if file_name.endswith('.pt')
        )
        if not shard_files:
            raise ValueError(f'未在 {val_dir} 找到验证分片，请先重新运行 image_processor_val.py')

        self.names = []
        self.hr_data = []
        self.lr_data = []

        for file_path in shard_files:
            packed = torch.load(file_path, weights_only=False)
            lr_key = f'lr_x{scale}'
            if not isinstance(packed, dict) or 'hr' not in packed or lr_key not in packed or 'names' not in packed:
                raise ValueError(f'{file_path} 不是新的验证分片格式')

            hr_batch = packed['hr']
            lr_batch = packed[lr_key]
            names = packed['names']

            if len(hr_batch) != len(lr_batch) or len(hr_batch) != len(names):
                raise ValueError(f'{file_path} 中 names/HR/LR 数量不一致')

            self.names.extend(names)
            self.hr_data.extend(list(hr_batch))
            self.lr_data.extend(list(lr_batch))

        if len(self.names) != len(set(self.names)):
            raise ValueError(f'{val_dir} 存在重复图片名')
    
    def __len__(self):
        return len(self.hr_data)
    
    def __getitem__(self, idx):
        return self.lr_data[idx], self.hr_data[idx]

def create_train_loader(datasets_dir, scale=2, batch_size=64, num_workers=8, patch_size=0):
    dataset = SRTrainDataset(datasets_dir, scale=scale, patch_size=patch_size)
    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    return train_loader

def create_val_loader(datasets_dir, scale=2):
    val_dir = os.path.join(datasets_dir)
    dataset = SRValDataset(val_dir, scale)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    return val_loader