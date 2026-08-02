import argparse
import copy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from models import BaselineSR, MDPSR
from utils import (
    MixedLoss,
    WarmupCosineScheduler,
    bicubic_metrics,
    create_logger,
    create_train_loader,
    create_val_loader,
    validate_metrics,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEACHER_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "BaselineSR_x2_0730_1753.pth"
DEFAULT_TRAIN_ROOT = Path("/home/tyzheng/Datasets_pt/train")
DEFAULT_VAL_ROOT = Path("/home/tyzheng/Datasets_pt/val")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill a frozen BaselineSR teacher into an MDPSR student."
    )
    parser.add_argument("--scale", type=int, default=2, choices=(2, 3, 4))
    parser.add_argument("--channel-nums", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=5)
    parser.add_argument(
        "--mixed-blocks",
        type=int,
        default=3,
        help="1-based MDPSR bridge block that expands half-width features.",
    )
    parser.add_argument("--in-channels", type=int, default=3, choices=(1, 3))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--minlr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=Path, default=SCRIPT_DIR / "checkpoints")
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--val-root", type=Path, default=DEFAULT_VAL_ROOT)
    parser.add_argument(
        "--teacher-fp",
        type=Path,
        default=DEFAULT_TEACHER_CHECKPOINT,
        help="Frozen BaselineSR checkpoint.",
    )
    parser.add_argument(
        "--student-fp",
        type=Path,
        default=None,
        help="Optional MDPSR checkpoint for same-architecture initialization.",
    )
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=0.1,
        help="Maximum Smooth-L1 weight between student and teacher SR outputs.",
    )
    parser.add_argument(
        "--distill-warmup-epochs",
        type=int,
        default=20,
        help="Epochs trained with ground-truth supervision only.",
    )
    parser.add_argument(
        "--distill-ramp-epochs",
        type=int,
        default=20,
        help="Epochs used to linearly ramp to --distill-weight; 0 enables it immediately.",
    )
    return parser.parse_args()


def _load_state(model, checkpoint_path, device, expected_model_type):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_config", {}).get("model_type")
    if model_type is not None and model_type != expected_model_type:
        raise ValueError(
            f"{checkpoint_path} contains {model_type}, expected {expected_model_type}."
        )
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)


def _build_bicubic_base(lr_img, scale):
    return F.interpolate(
        lr_img, scale_factor=scale, mode="bicubic", align_corners=False
    ).round().clamp(0, 255) / 255.0


def _distill_weight(args, epoch):
    if epoch < args.distill_warmup_epochs:
        return 0.0
    if args.distill_ramp_epochs <= 0:
        return args.distill_weight
    progress = (epoch - args.distill_warmup_epochs + 1) / args.distill_ramp_epochs
    return args.distill_weight * min(1.0, progress)


def train_distillation_epoch(student, teacher, train_loader, loss_func, optimizer, device, epoch, args, ema=None):
    student.train()
    teacher.eval()
    kd_weight = _distill_weight(args, epoch)
    totals = {"loss": 0.0, "supervised": 0.0, "distill": 0.0}

    for lr_img, hr_img in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
        lr_img = lr_img.to(device).float()
        hr_img = hr_img.to(device).float()
        optimizer.zero_grad(set_to_none=True)

        model_input = lr_img / 255.0
        target_sr = hr_img / 255.0
        bicubic = _build_bicubic_base(lr_img, student.scale)
        student_sr = bicubic + student(model_input)
        supervised_loss = loss_func(student_sr, target_sr)

        with torch.no_grad():
            teacher_sr = teacher(model_input)
        distill_loss = F.smooth_l1_loss(student_sr, teacher_sr)
        total_loss = supervised_loss + kd_weight * distill_loss

        total_loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update()

        totals["loss"] += total_loss.item()
        totals["supervised"] += supervised_loss.item()
        totals["distill"] += distill_loss.item()

    return {key: value / len(train_loader) for key, value in totals.items()}, kd_weight


def validate_student(student, val_loaders, loss_func, device):
    """Validate only against GT; teacher output never affects model selection."""
    student.eval()
    weighted_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for loader in val_loaders.values():
            loader_loss = 0.0
            for lr_img, hr_img in tqdm(loader, desc="loss-validating", leave=False):
                lr_img = lr_img.to(device).float()
                hr_img = hr_img.to(device).float()
                student_sr = _build_bicubic_base(lr_img, student.scale) + student(lr_img / 255.0)
                loader_loss += loss_func(student_sr, hr_img / 255.0).item()
            loader_loss /= len(loader)
            sample_count = len(loader.dataset)
            weighted_loss += loader_loss * sample_count
            total_samples += sample_count
    return weighted_loss / total_samples


def log_student_metrics(logger, student, val_loaders, args, device):
    for dataset_name, loader in val_loaders.items():
        logger.log_validation_results(
            dataset_name,
            validate_metrics(
                student, loader, args.scale, device, is_residual=True
            ),
        )


def main():
    args = parse_args()
    if args.distill_weight < 0:
        raise ValueError("--distill-weight must be non-negative")
    if args.distill_warmup_epochs < 0 or args.distill_ramp_epochs < 0:
        raise ValueError("distillation warmup and ramp epochs must be non-negative")
    if not args.teacher_fp.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_fp}")
    if args.student_fp is not None and not args.student_fp.is_file():
        raise FileNotFoundError(f"Student checkpoint not found: {args.student_fp}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    teacher = BaselineSR(
        args.scale, args.in_channels, args.channel_nums, args.num_blocks, bias=False
    ).to(device)
    _load_state(teacher, args.teacher_fp, device, "BaselineSR")
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = MDPSR(
        scale=args.scale,
        in_dim=args.in_channels,
        fea_dim=args.channel_nums,
        num_blocks=args.num_blocks,
        mixed_blocks=args.mixed_blocks,
        bias=False,
    ).to(device)
    if args.student_fp is not None:
        _load_state(student, args.student_fp, device, "MDPSR")

    train_loader = create_train_loader(
        args.train_root,
        scale=args.scale,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
    )
    val_loaders = {
        name: create_val_loader(args.val_root / name, args.scale, in_channels=args.in_channels)
        for name in ("Set5", "Set14", "B100", "U100", "M109")
    }

    timestamp = datetime.now().strftime("%m%d_%H%M")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.save_dir / f"MDPSR_distill_x{args.scale}_{timestamp}.pth"
    logger = create_logger("./logs", "MDPSR_distill", args.scale)
    logger.info(f"Using device: {device}")
    logger.info(f"Teacher: {args.teacher_fp}")
    logger.info(f"Student: MDPSR (mixed_blocks={args.mixed_blocks})")

    loss_func = MixedLoss(eps=1e-8, gamma=0.25)
    optimizer = optim.Adam(student.parameters(), betas=(0.9, 0.999), lr=args.lr)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        eta_min=args.minlr,
        warmup_start_lr=3e-4,
    )
    ema = ExponentialMovingAverage(student.parameters(), decay=args.ema_decay)
    logger.log_training_start(args, student.param_num(), len(train_loader))

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_stats, kd_weight = train_distillation_epoch(
            student, teacher, train_loader, loss_func, optimizer, device, epoch, args, ema
        )
        logger.log_epoch_train(epoch, args.epochs, train_stats["loss"], optimizer.param_groups[0]["lr"])
        logger.info(
            f"Epoch {epoch + 1}: supervised={train_stats['supervised']:.6f}, "
            f"distill={train_stats['distill']:.6f}, distill_weight={kd_weight:.4f}"
        )

        best_student = None
        with ema.average_parameters():
            val_loss = validate_student(student, val_loaders, loss_func, device)
            logger.log_epoch_val(epoch, args.epochs, val_loss)
            log_student_metrics(logger, student, val_loaders, args, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_student = copy.deepcopy(student)

        if best_student is not None:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "iteration": (epoch + 1) * len(train_loader),
                    "model_state_dict": best_student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "validation_loss": val_loss,
                    "model_config": {
                        "model_type": "MDPSR",
                        "is_mixed": True,
                        "mixed_blocks": args.mixed_blocks,
                        "teacher_model_type": "BaselineSR",
                        "teacher_checkpoint": str(args.teacher_fp),
                        "distill_weight": args.distill_weight,
                        "distill_warmup_epochs": args.distill_warmup_epochs,
                        "distill_ramp_epochs": args.distill_ramp_epochs,
                    },
                },
                checkpoint_path,
            )
            logger.log_best_model(val_loss)
        scheduler.step()

    logger.log_training_finished()
    best_student = MDPSR(
        scale=args.scale,
        in_dim=args.in_channels,
        fea_dim=args.channel_nums,
        num_blocks=args.num_blocks,
        mixed_blocks=args.mixed_blocks,
        bias=False,
    ).to(device)
    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    best_student.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    best_student.eval()
    logger.log_testing_start("Best MDPSR student")
    log_student_metrics(logger, best_student, val_loaders, args, device)
    logger.log_testing_start("Bicubic Interpolation")
    for dataset_name, loader in val_loaders.items():
        logger.log_validation_results(dataset_name, bicubic_metrics(loader, args.scale, device))
    logger.close()


if __name__ == "__main__":
    main()
