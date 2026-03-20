import torch
import torch.nn as nn
import torch.fft as fft
import math
from pytorch_msssim import ssim as pssim

def calculate_psnr(img1, img2):

    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'

    if img1.shape[0] == 3:
        img1 = to_y_channel(img1).to(torch.float32)
        img2 = to_y_channel(img2).to(torch.float32)
    elif img1.shape[0] == 1:
        img1 = img1[0].to(torch.float32)
        img2 = img2[0].to(torch.float32)
    else:
        raise ValueError(f'仅支持 1 或 3 通道，当前通道数: {img1.shape[0]}')

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10. * torch.log10(255.0 ** 2 / mse)
    
    return psnr


def calculate_ssim(img1, img2):

    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'

    if img1.shape[0] == 3:
        img1 = to_y_channel(img1).unsqueeze(0).unsqueeze(0).to(torch.float32)
        img2 = to_y_channel(img2).unsqueeze(0).unsqueeze(0).to(torch.float32)
    elif img1.shape[0] == 1:
        img1 = img1.unsqueeze(0).to(torch.float32)
        img2 = img2.unsqueeze(0).to(torch.float32)
    else:
        raise ValueError(f'仅支持 1 或 3 通道，当前通道数: {img1.shape[0]}')

    ssim = pssim(img1, img2, data_range=255.0)

    return ssim.item()

def to_y_channel(img):

    img = (img[0, :, :] * 65.481 + img[1, :, :] * 128.553 + img[2, :, :] * 24.966 + 16.0) / 255.0
    
    return img

class MixedLoss(nn.Module):

    def __init__(self, eps=1e-8, gamma=0.5):
        super().__init__()
        self.eps = eps
        self.gamma = float(gamma)

    def forward(self, pred, target):
        loss1 = self.Charbonnier_Loss(pred, target)
        loss2 = 0.0 if self.gamma == 0.0 else self.gamma * self.Frequency_Loss(pred, target)
        return loss1 + loss2

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

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=True):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10.0 / math.log(10.0)
        self.toY = toY
        self.register_buffer('coef', torch.tensor([65.481, 128.553, 24.966], dtype=torch.float32).reshape(1, 3, 1, 1))

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            coef = self.coef.to(pred.device)
            pred = (pred * coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred = pred / 255.
            target = target / 255.
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()