import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import metrics


def _build_residual_target(hr_img, lr_img, scale):
    base = F.interpolate(
        lr_img, scale_factor=scale, mode="bicubic", align_corners=False
    ).round().clamp(0, 255)
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
        metrics_list.sort(key=lambda item: item[0], reverse=True)
        metrics_list = metrics_list[: max(1, int(len(metrics_list) * clip_ratio))]
    return {
        "psnr": sum(item[0] for item in metrics_list) / len(metrics_list),
        "ssim": sum(item[1] for item in metrics_list) / len(metrics_list),
    }


def _evaluate_metrics_loop(val_loader, device, sr_builder, scale, clip_ratio, desc):
    metrics_list = []
    with torch.no_grad():
        for lr_img, hr_img in tqdm(val_loader, desc=desc, leave=False):
            lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
            sr_img = sr_builder(lr_img).round().clamp(0, 255)
            sr_img, hr_img = _crop_pair(sr_img, hr_img, scale)
            metrics_list.append(
                (
                    metrics.calculate_psnr(sr_img.squeeze(0), hr_img.squeeze(0)),
                    metrics.calculate_ssim(sr_img.squeeze(0), hr_img.squeeze(0)),
                )
            )
    return _finalize_metrics(metrics_list, clip_ratio)


def train_epoch(model, train_loader, loss_func, optimizer, device, epoch, ema=None, is_residual=True):
    """Train one full DPSR or one MDPSR model for an epoch."""
    model.train()
    running_loss = 0.0
    for lr_img, hr_img in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
        lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
        optimizer.zero_grad(set_to_none=True)

        target = (
            _build_residual_target(hr_img, lr_img, model.scale)
            if is_residual
            else hr_img / 255.0
        )
        prediction = model(lr_img / 255.0)
        loss = loss_func(prediction, target)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def validate_epoch(model, val_loader, loss_func, device, is_residual=True):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for lr_img, hr_img in tqdm(val_loader, desc="loss-validating", leave=False):
            lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
            target = (
                _build_residual_target(hr_img, lr_img, model.scale)
                if is_residual
                else hr_img / 255.0
            )
            val_loss += loss_func(model(lr_img / 255.0), target).item()
    return val_loss / len(val_loader)


def validate_metrics(model, val_loader, scale, device, clip_ratio=1.0, is_residual=True):
    model.eval()
    if is_residual:
        sr_builder = lambda lr_img: (
            model(lr_img / 255.0) * 255.0
            + F.interpolate(
                lr_img, scale_factor=model.scale, mode="bicubic", align_corners=False
            ).round().clamp(0, 255)
        )
    else:
        sr_builder = lambda lr_img: model(lr_img / 255.0) * 255.0
    return _evaluate_metrics_loop(
        val_loader, device, sr_builder, scale, clip_ratio, "metric-validating"
    )


def bicubic_metrics(val_loader, scale, device):
    return _evaluate_metrics_loop(
        val_loader,
        device,
        lambda lr_img: F.interpolate(
            lr_img, scale_factor=scale, mode="bicubic", align_corners=False
        ),
        scale,
        1.0,
        "bicubic-metrics-validating",
    )
