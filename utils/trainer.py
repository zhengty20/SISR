import torch
from tqdm import tqdm
from utils import metrics
from models import bilinear_interpolation
import torch.nn.functional as F


def _build_residual_target(hr_img, lr_img, scale):
    base = F.interpolate(lr_img, scale_factor=scale, mode='bilinear', align_corners=False).round().clamp(0, 255)
    return (hr_img - base) / 255.0


def _crop_pair(sr_img, hr_img, scale):
    crop_border = int(scale)
    if crop_border <= 0:
        return sr_img, hr_img
    return (
        sr_img[:, :, crop_border:-crop_border, crop_border:-crop_border],
        hr_img[:, :, crop_border:-crop_border, crop_border:-crop_border],
    )


def _finalize_metrics(metrics_list, clip_ratio):
    if clip_ratio < 1.0:
        metrics_list.sort(key=lambda x: x[0], reverse=True)
        selected_count = max(1, int(len(metrics_list) * clip_ratio))
        metrics_list = metrics_list[:selected_count]

    psnr_list = [item[0] for item in metrics_list]
    ssim_list = [item[1] for item in metrics_list]
    return {
        'psnr': sum(psnr_list) / len(psnr_list),
        'ssim': sum(ssim_list) / len(ssim_list)
    }


def _evaluate_metrics_loop(val_loader, device, sr_builder, scale, clip_ratio, desc):
    metrics_list = []
    with torch.no_grad():
        vpbar = tqdm(val_loader, desc=desc, leave=False)
        for lr_img, hr_img in vpbar:
            lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
            sr_img = sr_builder(lr_img).round().clamp(0, 255)
            sr_img, hr_img = _crop_pair(sr_img, hr_img, scale)
            psnr = metrics.calculate_psnr(sr_img.squeeze(0), hr_img.squeeze(0))
            ssim = metrics.calculate_ssim(sr_img.squeeze(0), hr_img.squeeze(0))
            metrics_list.append((psnr, ssim))
    return _finalize_metrics(metrics_list, clip_ratio)

def train_epoch(model, train_loader, loss_func, optimizer, device, epoch, ema=None, subnet_channels=16, shared_full_epochs=5):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False)
    for step, (lr_img, hr_img) in enumerate(pbar):
       
        lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        hr_img = _build_residual_target(hr_img, lr_img, model.scale)
        sr_img = model(lr_img / 255.)
        loss_full = loss_func(sr_img, hr_img)
        loss = loss_full

        if epoch < shared_full_epochs:
            loss = loss_full
        else:
            sr_sub = model.forward_shared_channel(lr_img / 255., subnet_channels)
            loss_sub = loss_func(sr_sub, hr_img)
            if step % 2 == 0:
                loss = loss_full
            else:
                loss = loss_sub
        
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update()
        
        running_loss += loss.item()

    return running_loss / len(train_loader)

def validate_epoch(model, val_loader, loss_func, device):
    """验证一个epoch，随机采样指定数量的图片"""
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        vpbar = tqdm(val_loader, desc='loss-validating', leave=False)
        for lr_img, hr_img in vpbar:
                
            lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
            hr_img = _build_residual_target(hr_img, lr_img, model.scale)
            sr_img = model(lr_img / 255.) 
            loss = loss_func(sr_img, hr_img)
            val_loss += loss.item()
            
    return val_loss / len(val_loader)

def validate_metrics(model, val_loader, scale, device, clip_ratio=1.0):
    model.eval()
    return _evaluate_metrics_loop(
        val_loader=val_loader,
        device=device,
        scale=scale,
        clip_ratio=clip_ratio,
        desc='metric-validating',
        sr_builder=lambda lr_img: (
            model(lr_img / 255.) * 255.
            + F.interpolate(lr_img, scale_factor=model.scale, mode='bilinear', align_corners=False)
        ),
    )

def validate_metrics_shared_channel(model, val_loader, scale, device, active_channels, clip_ratio=1.0):
    model.eval()
    if not hasattr(model, "forward_shared_channel"):
        raise AttributeError("模型不支持 forward_shared_channel")
    return _evaluate_metrics_loop(
        val_loader=val_loader,
        device=device,
        scale=scale,
        clip_ratio=clip_ratio,
        desc='metric-validating-shared',
        sr_builder=lambda lr_img: (
            model.forward_shared_channel(lr_img / 255., active_channels) * 255.
            + F.interpolate(lr_img, scale_factor=model.scale, mode='bilinear', align_corners=False)
        ),
    )

def bicubic_metrics(val_loader, scale, device):  
    return _evaluate_metrics_loop(
        val_loader=val_loader,
        device=device,
        scale=scale,
        clip_ratio=1.0,
        desc='basic-metrics-validating',
        sr_builder=lambda lr_img: F.interpolate(
            lr_img,
            scale_factor=scale,
            mode='bicubic',
            align_corners=False,
        ),
    )