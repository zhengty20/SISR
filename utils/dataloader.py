import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import kornia.augmentation as K

class SRDataset(Dataset):
    """超分辨率数据集类"""
    
    def __init__(self, hr_dir, lr_dir, scale=2):
        """
        Args:
            hr_dir: 高分辨率图像文件夹路径
            lr_dir: 低分辨率图像文件夹路径  
            scale: 超分倍数
            patch_size: 训练时的patch大小
            augment: 是否进行数据增强
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        
        # 获取图像文件列表
        self.hr_files = self._get_image_files(hr_dir)
        self.lr_files = self._get_image_files(lr_dir)
        
        # 确保HR和LR文件数量一致
        assert len(self.hr_files) == len(self.lr_files), \
            f"HR图像数量({len(self.hr_files)})与LR图像数量({len(self.lr_files)})不匹配"
    
    def _get_image_files(self, directory):
        """获取目录下的图像文件列表"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        files = []
        
        for ext in extensions:
            files.extend([f for f in os.listdir(directory) 
                         if f.lower().endswith(ext.lower())])
        
        files.sort()
        return files
    
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        # 读取图像
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        
        hr_img = imread_uint(hr_path)  # RGB format
        lr_img = imread_uint(lr_path)  # RGB format

        lr_tensor = np2tensor(lr_img)
        hr_tensor = np2tensor(hr_img)
        return lr_tensor, hr_tensor

def find_dataset_folders(datasets_dir, scale):
    """
    查找数据集文件夹
    
    Args:
        datasets_dir: 数据集根目录
        scale: 超分倍数
        
    Returns:
        dataset_folders: 数据集文件夹列表
    """
    dataset_folders = []
    
    # 查找HR和LR文件夹
    hr_dir = os.path.join(datasets_dir, 'HR')
    lr_dir = os.path.join(datasets_dir, f'LRbicx{scale}')
    
    # 检查数据集是否存在
    if os.path.exists(hr_dir) and os.path.exists(lr_dir):
        dataset_folders={
            'name': os.path.basename(datasets_dir),
            'hr_dir': hr_dir,
            'lr_dir': lr_dir
        }
    
    return dataset_folders

def create_dataloaders(datasets_dir, scale=3, batch_size=64, num_workers=16, shuffle=True, drop_last=True):
    """
    创建数据加载器
    
    Args:
        datasets_dir: 数据集根目录
        scale: 超分倍数  
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        shuffle: 是否打乱数据顺序
        drop_last: 是否丢弃最后不完整的批次
        
    Returns:
        data_loader: 数据加载器
    """
    # 查找数据集
    dataset_folders = find_dataset_folders(datasets_dir, scale)
    
    if not dataset_folders:
        raise ValueError(f"未找到X{scale}数据集!")
    
    # 合并所有数据集

    dataset = SRDataset(
        hr_dir=dataset_folders['hr_dir'],
        lr_dir=dataset_folders['lr_dir'],
        scale=scale
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 
    )
    
    return data_loader

def imread_uint(path):
    '''
    input: path
    output: HxWx3(RGB)
    '''

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

def np2tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)

class SRTrainDataset(Dataset):
    def __init__(self, hr_dir):

        self.hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith('.pt')])
        self.hr_data = []

        for i in range(len(self.hr_files)):
            x = torch.load(self.hr_files[i])
            self.hr_data.extend(x)
        
        self.hr_data = self.hr_data
    
    def __len__(self):
        return len(self.hr_data)
    
    def __getitem__(self, idx):
        return self.hr_data[idx]

class SRValDataset(Dataset):
    def __init__(self, val_dir, scale):

        self.hr_files = sorted([os.path.join(val_dir, 'HR', f) for f in os.listdir(os.path.join(val_dir, 'HR')) if f.endswith('.pt')])
        self.lr_files = sorted([os.path.join(val_dir, f'LR_bicubic_x{scale}', f) for f in os.listdir(os.path.join(val_dir, f'LR_bicubic_x{scale}')) if f.endswith('.pt')])
    
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        hr = torch.load(self.hr_files[idx])    
        lr = torch.load(self.lr_files[idx])
        return lr, hr

def create_train_loader(datasets_dir, batch_size=64, num_workers=8):
    hr_dir = os.path.join(datasets_dir, 'HR')
    dataset = SRTrainDataset(hr_dir)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                               num_workers=num_workers)
    return train_loader

def create_val_loader(datasets_dir, scale=2):
    val_dir = os.path.join(datasets_dir)
    dataset = SRValDataset(val_dir, scale)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    return val_loader

class SRKorniaAugmentor(torch.nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        self.augment = torch.nn.Sequential(
            K.RandomCrop(size=(144, 144)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation90(times=(0,3), p=0.5)
        )
        
        self.lr_only_aug = torch.nn.Sequential(
            K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.0), p=1.0),
            K.RandomGaussianNoise(mean=0.0, std=0.098, p=1.0)
        )

    def forward(self, hr):
        hr = self.augment(hr)
        lr = F.interpolate(hr, scale_factor=1/self.scale, mode='bicubic', align_corners=False)
        # lr = self.lr_only_aug(lr)

        return lr, hr