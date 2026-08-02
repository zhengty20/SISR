import copy
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch_ema import ExponentialMovingAverage

from models import BaselineSR, DPSR
from utils import (
    MixedLoss,
    WarmupCosineScheduler,
    bicubic_metrics,
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
            subnet_expand_block=args.subnet_expand_block,
        ).to(device)
        print(f"Using DPSR model for scale {args.scale}.")
        model_name = "DPSR"
    else:
        model = BaselineSR(
            scale=args.scale,
            in_dim=args.in_channels,
            fea_dim=args.channel_nums,
            num_blocks=args.num_blocks,
            bias=False,
        ).to(device)
        print(f"Using BaselineSR model for scale {args.scale}.")
        model_name = "BaselineSR"
    return model, model_name


def _load_pretrained_model(model, checkpoint_path, device, logger):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(model_state, strict=True)
    logger.info(f"Loaded full-width checkpoint: {checkpoint_path}")


def _model_configurations(args):
    """Return all depth/width executions used for validation and test metrics."""
    configurations = []
    active_depths = args.block_nums if args.is_residual else (None,)
    for active_num_blocks in active_depths:
        depth_name = f"blocks-{active_num_blocks}" if active_num_blocks else "blocks-full"
        configurations.append((f"{depth_name}/full", active_num_blocks, None, 1.0))
        if args.joint_width_training:
            configurations.append(
                (
                    f"{depth_name}/subnet-{args.subnet_width_mult:.1f}-x{args.subnet_expand_block}",
                    active_num_blocks,
                    args.subnet_width_mult,
                    args.subnet_loss_weight,
                )
            )
    return configurations


def _weighted_val_loss(
    model, val_loaders, loss_func, args, device, active_num_blocks=None, width_mult=None
):
    weighted_val_loss = 0.0
    total_val_samples = 0
    for loader in val_loaders.values():
        loader_loss = validate_epoch(
            model,
            loader,
            loss_func,
            device,
            is_residual=args.is_residual,
            width_mult=width_mult,
            active_num_blocks=active_num_blocks,
        )
        sample_count = len(loader.dataset)
        weighted_val_loss += loader_loss * sample_count
        total_val_samples += sample_count
    return weighted_val_loss / total_val_samples


def _validate_configurations(model, val_loaders, loss_func, args, device):
    configuration_losses = {}
    weighted_loss = 0.0
    total_weight = 0.0
    for name, active_num_blocks, width_mult, weight in _model_configurations(args):
        loss = _weighted_val_loss(
            model,
            val_loaders,
            loss_func,
            args,
            device,
            active_num_blocks=active_num_blocks,
            width_mult=width_mult,
        )
        configuration_losses[name] = loss
        weighted_loss += weight * loss
        total_weight += weight
    return {"combined": weighted_loss / total_weight, "configurations": configuration_losses}


def _log_model_metrics(logger, model, val_loaders, args, device):
    for config_name, active_num_blocks, width_mult, _ in _model_configurations(args):
        for dataset_name, loader in val_loaders.items():
            val_metrics = validate_metrics(
                model,
                loader,
                args.scale,
                device,
                1.0,
                is_residual=args.is_residual,
                width_mult=width_mult,
                active_num_blocks=active_num_blocks,
            )
            logger.log_validation_results(f"{dataset_name}/{config_name}", val_metrics)


def _normalize_block_configuration(args):
    args.block_nums = tuple(sorted(set(int(depth) for depth in args.block_nums)))
    if not args.block_nums:
        raise ValueError("--block_nums must contain at least one active depth")
    if any(depth < 1 or depth > args.num_blocks for depth in args.block_nums):
        raise ValueError(
            f"--block_nums must be in [1, {args.num_blocks}], got {args.block_nums}"
        )
    if args.is_residual and max(args.block_nums) != args.num_blocks:
        raise ValueError(
            "The largest --block_nums value must equal --num_blocks so every "
            "DPSR block is trained."
        )
    if not args.is_residual and args.block_nums != (args.num_blocks,):
        raise ValueError(
            "Multi-depth training is implemented for DPSR only; use "
            f"--block_nums {args.num_blocks} for BaselineSR."
        )
    if args.joint_width_training and args.subnet_expand_block > min(args.block_nums):
        raise ValueError(
            "--subnet_expand_block must not exceed the smallest active depth "
            "when training the half-width subnet."
        )


def main():
    args = train_parser()
    _normalize_block_configuration(args)
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
    logger.info(f"Joint active block depths: {args.block_nums}")

    if args.pretrained_fp:
        _load_pretrained_model(model, args.pretrained_fp, device, logger)

    train_loader = create_train_loader(
        "/home/tyzheng/Datasets_pt/train",
        scale=args.scale,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
    )
    val_loaders = {
        name: create_val_loader(
            f"/home/tyzheng/Datasets_pt/val/{name}",
            args.scale,
            in_channels=args.in_channels,
        )
        for name in ("Set5", "Set14", "B100", "U100", "M109")
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
            subnet_width_mult=args.subnet_width_mult,
            subnet_loss_weight=args.subnet_loss_weight,
            distill_loss_weight=args.distill_loss_weight,
            active_num_blocks=args.block_nums if args.is_residual else None,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        logger.log_epoch_train(epoch, args.epochs, train_loss, current_lr)

        best_candidate = None
        with ema.average_parameters():
            val_losses = _validate_configurations(model, val_loaders, loss_func, args, device)
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
                        "block_nums": list(args.block_nums),
                        "subnet_expand_block": args.subnet_expand_block,
                        "subnet_width_mult": args.subnet_width_mult,
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

    logger.log_testing_start("Bicubic Interpolation")
    for dataset_name, loader in val_loaders.items():
        logger.log_validation_results(
            dataset_name, bicubic_metrics(loader, args.scale, device)
        )
    logger.close()


if __name__ == "__main__":
    main()
