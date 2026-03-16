import torch
import torch.nn as nn
import torch.fft as fft
from pytorch_msssim import ssim as pssim

def calculate_psnr(img1, img2):

    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'

    img1 = to_y_channel(img1).clamp(0, 255).round().to(torch.float64)
    img2 = to_y_channel(img2).clamp(0, 255).round().to(torch.float64)

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10. * torch.log10(255.0 ** 2 / mse)
    
    return psnr


def calculate_ssim(img1, img2):

    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'

    img1 = to_y_channel(img1).unsqueeze(0).unsqueeze(0).clamp(0, 255).round().to(torch.float64)
    img2 = to_y_channel(img2).unsqueeze(0).unsqueeze(0).clamp(0, 255).round().to(torch.float64)

    ssim = pssim(img1, img2, data_range=255.0)

    return ssim.item()

def to_y_channel(img):

    img = (img[0, :, :] * 65.481 + img[1, :, :] * 128.553 + img[2, :, :] * 24.966 + 16.0) / 255.0
    
    return img

class MixedLoss(nn.Module):

    def __init__(self, eps=1e-8, gamma=0.5):
        super().__init__()
        self.eps = eps
        self.gamma = gamma

    def forward(self, pred, target):
        loss1 = self.Charbonnier_Loss(pred, target)
        loss2 = self.Frequency_Loss(pred, target)
        return loss1 + self.gamma *  loss2

    def Charbonnier_Loss(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff.pow(2) + self.eps)
        return loss.mean()

    def Frequency_Loss(self, pred, target):
        # 确保在执行FFT前为float32类型
        pred_f = pred.float()
        target_f = target.float()
        
        pred_fft = fft.fft2(pred_f, norm='ortho')
        target_fft = fft.fft2(target_f, norm='ortho')
        loss = torch.abs(pred_fft - target_fft)
        return loss.mean()