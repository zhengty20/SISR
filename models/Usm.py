import torch
import torch.nn.functional as F
        
def gaussian_blur2d(input, kernel_size, sigma):
    """
    Args:
        input (torch.Tensor): 输入张量，形状为 (B, C, H, W) 或 (C, H, W)
        kernel_size (tuple): 卷积核大小 (kh, kw)，要求kh == kw
        sigma (tuple): 高斯标准差 (sigma_x, sigma_y)，要求sigma_x == sigma_y
        border_type (str): 边界填充方式，支持 'reflect', 'replicate', 'constant'
    
    Returns:
        torch.Tensor: 高斯模糊后的张量
    """
    
    _, channels, _, _ = input.shape
    bits = 4
    
    # 计算padding
    pad = kernel_size // 2
    input_padded = F.pad(input, (pad, pad, pad, pad), mode='reflect')
    
    # 水平方向卷积
    # kernel_1d = create_gaussian_kernel_1d(kernel_size, sigma, bits).to(input.device)
    # print(kernel_1d)
    kernel_1d = torch.tensor([2., 4., 4., 4., 2.]).to(input.device)
    kernel_1d = kernel_1d.view(1, 1, kernel_size, 1).expand(channels, 1, kernel_size, 1)
    output = F.conv2d(input_padded, kernel_1d, groups=channels, padding=(0, 0))
    output = (output / (2 ** bits)).floor()

    
    # 垂直方向卷积
    kernel_1d_v = kernel_1d.transpose(-1, -2)
    output = F.conv2d(output, kernel_1d_v, groups=channels, padding=(0, 0))
    output = (output / (2 ** bits)).floor()

    return output

def create_gaussian_kernel_1d(kernel_size, sigma, bits=8):
    """创建1D高斯卷积核"""
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = (kernel / kernel.sum() * (2 ** bits)).round()
    return kernel 

def usm_interpolation(x, scale):
    
    x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False).floor()
    usm = gaussian_blur2d(x, 5, 2.0)
    x = (x + 9 / 16 * (x - usm)).floor().clamp(0, 255)
     
    return x