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


def _active_depths(active_num_blocks):
    if active_num_blocks is None:
        return (None,)
    if isinstance(active_num_blocks, int):
        return (active_num_blocks,)
    return tuple(active_num_blocks)


def _forward_model(model, model_input, width_mult=None, active_num_blocks=None):
    kwargs = {}
    if width_mult is not None:
        kwargs["width_mult"] = width_mult
    if active_num_blocks is not None:
        kwargs["active_num_blocks"] = active_num_blocks
    return model(model_input, **kwargs)


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
    subnet_width_mult=0.5,
    subnet_loss_weight=1.0,
    distill_loss_weight=0.0,
    active_num_blocks=None,
):
    """Train every requested active depth with shared DPSR parameters."""
    model.train()
    running_loss = 0.0
    active_depths = _active_depths(active_num_blocks)

    for lr_img, hr_img in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
        lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        target = (
            _build_residual_target(hr_img, lr_img, model.scale)
            if is_residual
            else hr_img / 255.0
        )
        model_input = lr_img / 255.0
        depth_losses = []

        for active_depth in active_depths:
            full_sr = _forward_model(
                model, model_input, active_num_blocks=active_depth
            )
            full_loss = loss_func(full_sr, target)
            depth_loss = full_loss

            if joint_width_training:
                subnet_sr = _forward_model(
                    model,
                    model_input,
                    width_mult=subnet_width_mult,
                    active_num_blocks=active_depth,
                )
                subnet_loss = loss_func(subnet_sr, target)
                depth_loss = (
                    full_loss + subnet_loss_weight * subnet_loss
                ) / (1.0 + subnet_loss_weight)
                if distill_loss_weight > 0.0:
                    depth_loss = depth_loss + distill_loss_weight * F.l1_loss(
                        subnet_sr, full_sr.detach()
                    )
            depth_losses.append(depth_loss)

        loss = sum(depth_losses) / len(depth_losses)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def validate_epoch(
    model,
    val_loader,
    loss_func,
    device,
    is_residual=True,
    width_mult=None,
    active_num_blocks=None,
):
    model.eval()
    val_loss = 0.0
    width_label = "full" if width_mult is None or width_mult == 1.0 else f"{width_mult:.1f}"
    depth_label = "all" if active_num_blocks is None else str(active_num_blocks)
    with torch.no_grad():
        for lr_img, hr_img in tqdm(
            val_loader, desc=f"loss-validating[b{depth_label}/{width_label}]", leave=False
        ):
            lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
            target = (
                _build_residual_target(hr_img, lr_img, model.scale)
                if is_residual
                else hr_img / 255.0
            )
            prediction = _forward_model(
                model,
                lr_img / 255.0,
                width_mult=width_mult,
                active_num_blocks=active_num_blocks,
            )
            val_loss += loss_func(prediction, target).item()
    return val_loss / len(val_loader)


def validate_metrics(
    model,
    val_loader,
    scale,
    device,
    clip_ratio=1.0,
    is_residual=True,
    width_mult=None,
    active_num_blocks=None,
):
    model.eval()
    model_forward = lambda x: _forward_model(
        model,
        x,
        width_mult=width_mult,
        active_num_blocks=active_num_blocks,
    )
    if is_residual:
        sr_builder = lambda lr_img: (
            model_forward(lr_img / 255.0) * 255.0
            + F.interpolate(
                lr_img, scale_factor=scale, mode="bicubic", align_corners=False
            ).round().clamp(0, 255)
        )
    else:
        sr_builder = lambda lr_img: model_forward(lr_img / 255.0) * 255.0
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
