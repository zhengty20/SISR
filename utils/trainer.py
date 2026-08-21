import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import channel_label
from utils import metrics


def _build_residual_target(hr_img, lr_img, scale):
    base = (
        F.interpolate(lr_img, scale_factor=scale, mode="bilinear", align_corners=False)
        .round()
        .clamp(0, 255)
    )
    return (hr_img - base) / 255.0


def _crop_pair(sr_img, hr_img, scale):
    border = int(scale)
    if border <= 0:
        return sr_img, hr_img
    return (
        sr_img[:, :, border:-border, border:-border],
        hr_img[:, :, border:-border, border:-border],
    )


def _finalize_metrics(metrics_list, clip_ratio):
    if clip_ratio < 1.0:
        metrics_list.sort(key=lambda item: item[0], reverse=True)
        metrics_list = metrics_list[: max(1, int(len(metrics_list) * clip_ratio))]
    return {
        "psnr": sum(item[0] for item in metrics_list) / len(metrics_list),
        "ssim": sum(item[1] for item in metrics_list) / len(metrics_list),
    }


def _forward_model(model, model_input, channels=None):
    return model(model_input, channels=channels)


def _evaluate_metrics_loop(val_loader, device, sr_builder, scale, clip_ratio, desc):
    metrics_list = []
    with torch.no_grad():
        for lr_img, hr_img in tqdm(val_loader, desc=desc, leave=False):
            lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
            sr_img, hr_img = _crop_pair(
                sr_builder(lr_img).round().clamp(0, 255), hr_img, scale
            )
            metrics_list.append(
                (
                    metrics.calculate_psnr(sr_img.squeeze(0), hr_img.squeeze(0)),
                    metrics.calculate_ssim(sr_img.squeeze(0), hr_img.squeeze(0)),
                )
            )
    return _finalize_metrics(metrics_list, clip_ratio)


def train_epoch(
    model,
    train_loader,
    loss_func,
    optimizer,
    device,
    epoch,
    ema=None,
    is_residual=True,
    joint_width_training=False,
    alternate_width_training=False,
    subnet_channels=16,
    subnet_loss_weight=1.0,
    distill_loss_weight=0.0,
):
    """Train full/subnet paths jointly or alternate one width per batch."""
    if joint_width_training and alternate_width_training:
        raise ValueError('joint and alternate width training are mutually exclusive')
    model.train()
    running_loss = 0.0
    for batch_index, (lr_img, hr_img) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
    ):
        lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        target = (
            _build_residual_target(hr_img, lr_img, model.scale)
            if is_residual
            else hr_img / 255.0
        )
        model_input = lr_img / 255.0
        if alternate_width_training:
            train_subnet = (batch_index + epoch) % 2 == 1
            active_channels = subnet_channels if train_subnet else model.fea_dim
            sr_img = _forward_model(model, model_input, channels=active_channels)
            loss = loss_func(sr_img, target)
            if train_subnet:
                loss = subnet_loss_weight * loss
        else:
            full_sr = _forward_model(model, model_input, channels=model.fea_dim)
            full_loss = loss_func(full_sr, target)
            loss = full_loss
        if joint_width_training:
            subnet_sr = _forward_model(model, model_input, channels=subnet_channels)
            subnet_loss = loss_func(subnet_sr, target)
            loss = (full_loss + subnet_loss_weight * subnet_loss) / (
                1.0 + subnet_loss_weight
            )
            if distill_loss_weight > 0.0:
                loss += distill_loss_weight * F.l1_loss(subnet_sr, full_sr.detach())
        loss.backward()
        optimizer.step()
        if hasattr(model, 'project_quantization_parameters'):
            model.project_quantization_parameters()
        if ema is not None:
            ema.update()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def validate_epoch(
    model, val_loader, loss_func, device, is_residual=True, channels=None
):
    model.eval()
    channels = model.fea_dim if channels is None else channels
    val_loss = 0.0
    with torch.no_grad():
        for lr_img, hr_img in tqdm(
            val_loader, desc=f"loss-validating[{channel_label(channels)}]", leave=False
        ):
            lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
            target = (
                _build_residual_target(hr_img, lr_img, model.scale)
                if is_residual
                else hr_img / 255.0
            )
            val_loss += loss_func(
                _forward_model(model, lr_img / 255.0, channels), target
            ).item()
    return val_loss / len(val_loader)


def validate_metrics(
    model, val_loader, scale, device, clip_ratio=1.0, is_residual=True, channels=None
):
    channels = model.fea_dim if channels is None else channels
    model.eval()
    model_forward = lambda x: _forward_model(model, x, channels)
    if is_residual:
        sr_builder = lambda lr_img: model_forward(
            lr_img / 255.0
        ) * 255.0 + F.interpolate(
            lr_img, scale_factor=scale, mode="bilinear", align_corners=False
        ).round().clamp(
            0, 255
        )
    else:
        sr_builder = lambda lr_img: model_forward(lr_img / 255.0) * 255.0
    return _evaluate_metrics_loop(
        val_loader, device, sr_builder, scale, clip_ratio, "metric-validating"
    )


def bilinear_metrics(val_loader, scale, device):
    return _evaluate_metrics_loop(
        val_loader,
        device,
        lambda lr_img: F.interpolate(
            lr_img, scale_factor=scale, mode="bilinear", align_corners=False
        ),
        scale,
        1.0,
        "bilinear-metrics-validating",
    )
