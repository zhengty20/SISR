import torch
from typing import Tuple, Dict
from torch import Tensor
from .Bilinear import bilinear_interpolation

class ARMSR:
    def __init__(self, patch_size: Tuple[int, int], scale_factor: int, overlap: int = 0, device: str = 'cuda'):
        """
        Args:
            patch_size: (height, width) 低分辨率patch尺寸
            scale_factor: 放大倍数
            overlap: 重叠像素数（基于低分辨率）
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.lr_patch_h, self.lr_patch_w = patch_size
        self.scale = scale_factor
        self.overlap = min(overlap, min(self.lr_patch_h, self.lr_patch_w))
        self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
        
        # 低分辨率步长
        self.lr_stride_h = self.lr_patch_h - self.overlap
        self.lr_stride_w = self.lr_patch_w - self.overlap
        
        # 高分辨率对应尺寸
        self.hr_patch_h = self.lr_patch_h * scale_factor
        self.hr_patch_w = self.lr_patch_w * scale_factor
        self.hr_overlap = self.overlap * scale_factor
        self.hr_stride_h = self.lr_stride_h * scale_factor
        self.hr_stride_w = self.lr_stride_w * scale_factor
    
    def split_image(self, lr_image: Tensor) -> Dict:
        """
        Args:
            lr_image: 低分辨率输入图像
                     支持格式: (C, H, W) 或 (H, W, C) 或 (H, W)
            
        Returns:
            包含patch信息的字典
        """
        channels, lr_h, lr_w = lr_image.shape
        
        # 计算patch网格
        n_patches_h = (lr_h - self.overlap + self.lr_stride_h - 1) // self.lr_stride_h
        n_patches_w = (lr_w - self.overlap + self.lr_stride_w - 1) // self.lr_stride_w
        
        # 预分配tensor存储所有patches
        total_patches = n_patches_h * n_patches_w
        lr_patches = torch.zeros(total_patches, channels, self.lr_patch_h, self.lr_patch_w, 
                                device=self.device, dtype=lr_image.dtype)
        
        # 预计算坐标信息
        patch_coords = []
        patch_idx = 0
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # 低分辨率patch位置
                lr_start_h = i * self.lr_stride_h
                lr_start_w = j * self.lr_stride_w
                
                # 边界处理
                lr_end_h = min(lr_start_h + self.lr_patch_h, lr_h)
                lr_end_w = min(lr_start_w + self.lr_patch_w, lr_w)
                
                # 调整边界patch位置确保完整尺寸
                if lr_end_h - lr_start_h < self.lr_patch_h:
                    lr_start_h = max(0, lr_h - self.lr_patch_h)
                    lr_end_h = lr_start_h + self.lr_patch_h
                if lr_end_w - lr_start_w < self.lr_patch_w:
                    lr_start_w = max(0, lr_w - self.lr_patch_w)
                    lr_end_w = lr_start_w + self.lr_patch_w
                
                # 对应的高分辨率位置
                hr_start_h = lr_start_h * self.scale
                hr_start_w = lr_start_w * self.scale
                hr_end_h = lr_end_h * self.scale
                hr_end_w = lr_end_w * self.scale
                
                # 提取patch
                lr_patch = lr_image[:, lr_start_h:lr_end_h, lr_start_w:lr_end_w]
                
                # 处理边界padding
                if lr_patch.shape[1] < self.lr_patch_h or lr_patch.shape[2] < self.lr_patch_w:
                    padded_patch = torch.zeros(channels, self.lr_patch_h, self.lr_patch_w, 
                                             device=self.device, dtype=lr_image.dtype)
                    padded_patch[:, :lr_patch.shape[1], :lr_patch.shape[2]] = lr_patch
                    lr_patch = padded_patch
                
                lr_patches[patch_idx] = lr_patch
                
                patch_coords.append({
                    'lr_pos': (lr_start_h, lr_start_w, lr_end_h, lr_end_w),
                    'hr_pos': (hr_start_h, hr_start_w, hr_end_h, hr_end_w),
                    'patch_id': patch_idx
                })
                
                patch_idx += 1
        
        return {
            'lr_patches': lr_patches,
            'patch_coords': patch_coords,
            'hr_shape': (channels, lr_h * self.scale, lr_w * self.scale)
        }
    
    def dynamic_processing(self, patch_data):
        """
        Args:
            patch_data: patch数据
            model: PyTorch模型或处理函数，接收(B, C, H, W)返回(B, C, H*scale, W*scale)
            batch_size: 批处理大小
            
        Returns:
            包含处理后高分辨率patches的数据
        """
        lr_patches = patch_data['lr_patches']
        total_patches = lr_patches.shape[0]
        channels = lr_patches.shape[1]
        
        hr_patches = torch.zeros(total_patches, channels, self.hr_patch_h, self.hr_patch_w, device=self.device, dtype=lr_patches.dtype)
        
        # 批量处理
        with torch.no_grad():
            for i in range(0, total_patches):
                lr = lr_patches[i : i + 1, :, :, :]
                hr_patches[i] = bilinear_interpolation(lr, self.scale)
        
        result = patch_data.copy()
        result['hr_patches'] = hr_patches
        return result
    
    def reconstruct_hr_image(self, patch_data: Dict):
        """
        Args:
            patch_data: 包含hr_patches的数据
            
        Returns:
            重构的高分辨率图像
        """
        
        hr_patches = patch_data['hr_patches']
        patch_coords = patch_data['patch_coords']
        channels, hr_h, hr_w = patch_data['hr_shape']
        
        # 初始化输出tensor
        hr_image = torch.zeros(channels, hr_h, hr_w, device=self.device, dtype=hr_patches.dtype)
        overlap_count = torch.zeros(hr_h, hr_w, device=self.device, dtype=torch.int8)
        
        # 合并所有patches
        for patch_idx, coord in enumerate(patch_coords):
            hr_start_h, hr_start_w, hr_end_h, hr_end_w = coord['hr_pos']
            hr_patch = hr_patches[patch_idx]
            
            # 处理边界情况
            actual_h = min(hr_end_h, hr_h) - hr_start_h
            actual_w = min(hr_end_w, hr_w) - hr_start_w
            
            if actual_h > 0 and actual_w > 0:
                # 直接累加patch值
                hr_image[:, hr_start_h : hr_start_h + actual_h, hr_start_w : hr_start_w + actual_w] += hr_patch[:, : actual_h, : actual_w]
                
                # 记录重叠次数
                overlap_count[hr_start_h : hr_start_h + actual_h, hr_start_w : hr_start_w + actual_w] += 1
        
        hr_image = hr_image / overlap_count.unsqueeze(0).float()

        return hr_image.unsqueeze(0)
    
    def full_pipeline(self, lr_image):

        """
        Args:
            lr_image: 低分辨率输入图像
            model: 可选的超分辨率模型
            
        Returns:
            (高分辨率图像, patch数据)
        """
        # 1. 切分图像
        patch_data = self.split_image(lr_image)
        
        # 2. 处理patches
        patch_data = self.dynamic_processing(patch_data)
        
        # 3. 重构图像
        hr_image = self.reconstruct_hr_image(patch_data)
        
        return hr_image
