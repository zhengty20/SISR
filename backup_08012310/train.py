import copy
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch_ema import ExponentialMovingAverage

from models import BaselineSR, DPSR, MDPSR
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
    if not args.is_residual:
        model = BaselineSR(
            scale=args.scale,
            in_dim=args.in_channels,
            fea_dim=args.channel_nums,
            num_blocks=args.num_blocks,
            bias=False,
        ).to(device)
        return model, "BaselineSR"

    if args.is_mixed:
        model = MDPSR(
            scale=args.scale,
            in_dim=args.in_channels,
            fea_dim=args.channel_nums,
            num_blocks=args.num_blocks,
            mixed_blocks=args.mixed_blocks,
            bias=False,
        ).to(device)
        return model, "MDPSR"

    model = DPSR(
        scale=args.scale,
        in_dim=args.in_channels,
        fea_dim=args.channel_nums,
        num_blocks=args.num_blocks,
        bias=False,
    ).to(device)
    return model, "DPSR"


def _load_pretrained_model(model, checkpoint_path, device, logger, model_name):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_model_name = checkpoint.get("model_config", {}).get("model_type")
    if saved_model_name is not None and saved_model_name != model_name:
        raise ValueError(
            f"Checkpoint model type is {saved_model_name}, but requested {model_name}."
        )
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    logger.info(f"Loaded {model_name} checkpoint: {checkpoint_path}")


def _weighted_val_loss(model, val_loaders, loss_func, args, device):
    weighted_loss = 0.0
    total_samples = 0
    for loader in val_loaders.values():
        loader_loss = validate_epoch(
            model, loader, loss_func, device, is_residual=args.is_residual
        )
        sample_count = len(loader.dataset)
        weighted_loss += loader_loss * sample_count
        total_samples += sample_count
    return weighted_loss / total_samples


def _log_model_metrics(logger, model, val_loaders, args, device):
    for dataset_name, loader in val_loaders.items():
        val_metrics = validate_metrics(
            model,
            loader,
            args.scale,
            device,
            1.0,
            is_residual=args.is_residual,
        )
        logger.log_validation_results(dataset_name, val_metrics)


def main():
    args = train_parser()
    if args.is_mixed and not args.is_residual:
        raise ValueError("--is_mixed requires --is_residual")

    print(f"Training configuration: {args}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, model_name = _build_model(args, device)

    logger = create_logger(log_dir="./logs", model_name=model_name, scale=args.scale)
    logger.info(f"Using device: {device}")
    logger.info(
        f"Using {model_name} model for scale {args.scale}"
        + (f" (mixed_blocks={args.mixed_blocks})" if args.is_mixed else ".")
    )

    if args.pretrained_fp:
        _load_pretrained_model(
            model, args.pretrained_fp, device, logger, model_name
        )

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

    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{model_name}_x{args.scale}_{timestamp}.pth"

    total_params = model.param_num()
    logger.info(f"Model parameters: {total_params:,}")
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
        )
        current_lr = optimizer.param_groups[0]["lr"]
        logger.log_epoch_train(epoch, args.epochs, train_loss, current_lr)

        best_candidate = None
        with ema.average_parameters():
            val_loss = _weighted_val_loss(model, val_loaders, loss_func, args, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_candidate = copy.deepcopy(model)
            logger.log_epoch_val(epoch, args.epochs, val_loss)
            _log_model_metrics(logger, model, val_loaders, args, device)

        if best_candidate is not None:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "iteration": (epoch + 1) * len(train_loader),
                    "model_state_dict": best_candidate.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "validation_loss": val_loss,
                    "model_config": {
                        "model_type": model_name,
                        "is_mixed": args.is_mixed,
                        "mixed_blocks": args.mixed_blocks if args.is_mixed else None,
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
    net.load_state_dict(state_dict["model_state_dict"], strict=True)
    net.eval()
    _log_model_metrics(logger, net, val_loaders, args, device)

    logger.log_testing_start("Bicubic Interpolation")
    for dataset_name, loader in val_loaders.items():
        logger.log_validation_results(dataset_name, bicubic_metrics(loader, args.scale, device))
    logger.close()


if __name__ == "__main__":
    main()
