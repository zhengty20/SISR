import argparse


def train_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--channel_nums", type=int, default=32, help="feature channel count")
    parser.add_argument("--num_blocks", type=int, default=5, help="number of blocks")
    parser.add_argument(
        "--is_mixed",
        action="store_true",
        help="train MDPSR instead of the full-width DPSR",
    )
    parser.add_argument(
        "--mixed_blocks",
        type=int,
        default=3,
        help="1-based MDPSR bridge block that expands half-width features",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--minlr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pretrained_fp", type=str, default="", help="same-architecture checkpoint")
    parser.add_argument("--wbits", type=int, default=8)
    parser.add_argument("--abits", type=int, default=8)
    parser.add_argument("--in_channels", type=int, default=3, choices=[1, 3])
    parser.add_argument("--patch_size", type=int, default=24)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--is_residual", action="store_true")
    return parser.parse_args()
