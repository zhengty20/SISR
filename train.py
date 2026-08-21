import copy
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch_ema import ExponentialMovingAverage

from models import FSRCNN, DPSR, channel_label
from utils import (
    MixedLoss,
    WarmupCosineScheduler,
    bilinear_metrics,
    create_logger,
    create_train_loader,
    create_val_loader,
    train_epoch,
    train_parser,
    validate_epoch,
    validate_metrics,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def _build_model(args, device):
    if args.is_residual:
        model = DPSR(
            scale=args.scale,
            in_dim=args.in_channels,
            fea_dim=args.channel_nums,
            num_blocks=args.num_blocks,
            bias=False,
            subnet_channels=args.subnet_channels,
        ).to(device)
        print(f"Using DPSR model for scale {args.scale}.")
        model_name = "DPSR"
    else:
        model = FSRCNN(
            scale_factor=args.scale,
            num_channels=args.in_channels
        ).to(device)
        print(f"Using FSRCNN model for scale {args.scale}.")
        model_name = "FSRCNN"
    return model, model_name

def _load_pretrained_model(model, checkpoint_path, device, logger):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(model_state, strict=True)
    logger.info(f"Loaded full-width checkpoint: {checkpoint_path}")


def _model_configurations(args):
    """Return the full-channel path and optional explicit subnet path."""
    configurations = [(channel_label(args.channel_nums), args.channel_nums)]
    if args.joint_width_training:
        configurations.append(
            (channel_label(args.subnet_channels), args.subnet_channels)
        )
    return configurations


def _weighted_val_loss(model, val_loaders, loss_func, args, device, channels):
    weighted_val_loss = 0.0
    total_val_samples = 0
    for loader in val_loaders.values():
        loader_loss = validate_epoch(
            model,
            loader,
            loss_func,
            device,
            is_residual=args.is_residual,
            channels=channels,
        )
        sample_count = len(loader.dataset)
        weighted_val_loss += loader_loss * sample_count
        total_val_samples += sample_count
    return weighted_val_loss / total_val_samples


def _validate_configurations(model, val_loaders, loss_func, args, device):
    configuration_losses = {}
    for name, channels in _model_configurations(args):
        configuration_losses[name] = _weighted_val_loss(
            model, val_loaders, loss_func, args, device, channels
        )
    full_loss = configuration_losses[channel_label(args.channel_nums)]
    subnet_loss = configuration_losses.get(channel_label(args.subnet_channels))
    combined_loss = full_loss
    if subnet_loss is not None:
        combined_loss = (full_loss + args.subnet_loss_weight * subnet_loss) / (
            1.0 + args.subnet_loss_weight
        )
    return {"combined": combined_loss, "configurations": configuration_losses}


def _log_model_metrics(logger, model, val_loaders, args, device):
    for config_name, channels in _model_configurations(args):
        for dataset_name, loader in val_loaders.items():
            val_metrics = validate_metrics(
                model,
                loader,
                args.scale,
                device,
                1.0,
                is_residual=args.is_residual,
                channels=channels,
            )
            logger.log_validation_results(f"{dataset_name}/{config_name}", val_metrics)


def main():
    args = train_parser()
    if args.joint_width_training and not args.is_residual:
        raise ValueError("--joint_width_training is only supported by DPSR")
    if args.subnet_loss_weight < 0.0:
        raise ValueError("--subnet_loss_weight must be non-negative")
    if args.distill_loss_weight < 0.0:
        raise ValueError("--distill_loss_weight must be non-negative")

    print(f"Training configuration: {args}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, model_name = _build_model(args, device)

    logger = create_logger(log_dir="./logs", model_name=model_name, scale=args.scale)
    logger.info(f"使用设备: {device}")
    logger.info(
        f"Full path: {channel_label(args.channel_nums)}; subnet path: "
        f"{channel_label(args.subnet_channels)}; blocks: {args.num_blocks}"
    )

    if args.pretrained_fp:
        _load_pretrained_model(model, args.pretrained_fp, device, logger)

    datasets_root = Path(args.datasets_root)
    train_loader = create_train_loader(
        datasets_root / "train",
        scale=args.scale,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
    )
    val_loaders = {
        dataset_name: create_val_loader(
            datasets_root / "val" / dataset_name,
            args.scale,
            in_channels=args.in_channels,
        )
        for dataset_name in ("Set5", "Set14", "B100", "U100", "M109")
    }

    time_stamp = datetime.now().strftime("%m%d_%H%M")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{model_name}_x{args.scale}_{time_stamp}.pth"

    total_params = model.param_num()
    logger.info(f"模型总参数量: {total_params:,}")
    loss_func = MixedLoss(eps=1e-8, gamma=0.25)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr)
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        eta_min=args.minlr,
        warmup_start_lr=3e-4,
    )

    logger.log_training_start(args, total_params, len(train_loader))
    best_val_loss = float("inf")
    logger.info("Begin Training")

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            loss_func,
            optimizer,
            device,
            epoch,
            ema=ema,
            is_residual=args.is_residual,
            joint_width_training=args.joint_width_training,
            subnet_channels=args.subnet_channels,
            subnet_loss_weight=args.subnet_loss_weight,
            distill_loss_weight=args.distill_loss_weight,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        logger.log_epoch_train(epoch, args.epochs, train_loss, current_lr)

        best_candidate = None
        with ema.average_parameters():
            val_losses = _validate_configurations(
                model, val_loaders, loss_func, args, device
            )
            val_loss = val_losses["combined"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_candidate = copy.deepcopy(model)

            logger.log_epoch_val(epoch, args.epochs, val_loss)
            logger.info(
                "Validation configuration losses: "
                + ", ".join(
                    f"{name}={loss:.6f}"
                    for name, loss in val_losses["configurations"].items()
                )
            )
            _log_model_metrics(logger, model, val_loaders, args, device)

        if best_candidate is not None:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "iteration": (epoch + 1) * len(train_loader),
                    "model_state_dict": best_candidate.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "validation_losses": val_losses["configurations"],
                    "model_config": {
                        "num_blocks": args.num_blocks,
                        "full_channels": args.channel_nums,
                        "subnet_channels": args.subnet_channels,
                    },
                },
                model_path,
            )
            logger.log_best_model(val_loss)

        scheduler.step()

    logger.log_training_finished()
    logger.log_testing_start("Best Model")
    net, _ = _build_model(args, device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    net.load_state_dict(state_dict["model_state_dict"])
    net.eval()
    _log_model_metrics(logger, net, val_loaders, args, device)

    logger.log_testing_start("Bilinear Interpolation")
    for dataset_name, loader in val_loaders.items():
        logger.log_validation_results(
            dataset_name, bilinear_metrics(loader, args.scale, device)
        )
    logger.close()


if __name__ == "__main__":
    main()
