import argparse


def train_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--scale", type=int, default=2, choices=[2, 3, 4], help="super-resolution scale"
    )
    parser.add_argument(
        "--channel_nums",
        type=int,
        default=32,
        choices=[32],
        help="full-path feature channels",
    )
    parser.add_argument(
        "--subnet-channels",
        type=int,
        default=16,
        help="explicit feature channels used by the subnet path",
    )
    parser.add_argument("--num_blocks", type=int, default=5, help="network block count")
    parser.add_argument(
        "--epochs", type=int, default=300, help="number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="initial learning rate")
    parser.add_argument(
        "--minlr", type=float, default=1e-5, help="minimum learning rate"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="data loader worker count"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./checkpoints", help="checkpoint directory"
    )
    parser.add_argument("--device", type=str, default="cuda", help="training device")
    parser.add_argument(
        "--pretrained_fp",
        type=str,
        default="",
        help="full-channel DPSR checkpoint used for initialization",
    )
    parser.add_argument(
        "--wbits", type=int, default=8, help="weight bitwidth for quantization"
    )
    parser.add_argument(
        "--abits", type=int, default=8, help="activation bitwidth for quantization"
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        choices=[1, 3],
        help="input channels, 1 for Y and 3 for RGB",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=24,
        help="HR crop size, 0 means choose automatically by scale",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=15, help="number of warmup epochs"
    )
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay")
    parser.add_argument(
        "--is_residual", action="store_true", help="whether to use residual learning"
    )
    parser.add_argument(
        "--joint_width_training",
        action="store_true",
        help="train the full and explicit subnet paths in every step",
    )
    parser.add_argument(
        "--subnet_loss_weight",
        type=float,
        default=1.0,
        help="weight of the subnet supervised loss",
    )
    parser.add_argument(
        "--distill_loss_weight",
        type=float,
        default=0.1,
        help="weight of L1 distillation from full output to subnet output",
    )
    return parser.parse_args()
