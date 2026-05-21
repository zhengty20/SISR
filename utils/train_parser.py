import argparse


def train_parser():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--model_name', type=str, default='DPSR', help='model name')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4], help='super-resolution scale')
    parser.add_argument('--channel_nums', type=int, default=32, help='feature channel count')
    parser.add_argument('--num_blocks', type=int, default=5, help='number of blocks')
    parser.add_argument('--epochs', type=int, default=300, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='initial learning rate')
    parser.add_argument('--minlr', type=float, default=1e-5, help='minimum learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='data loader worker count')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda', help='training device')
    parser.add_argument('--in_channels', type=int, default=3, choices=[1, 3], help='input channels, 1 for Y and 3 for RGB')
    parser.add_argument('--patch_size', type=int, default=0, help='HR crop size, 0 means choose automatically by scale')
    parser.add_argument('--warmup_epochs', type=int, default=15, help='number of warmup epochs')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay')
    parser.add_argument('--shared_subnet_channels', type=int, default=16, help='shared subnet channel count')
    parser.add_argument('--shared_full_epochs', type=int, default=50, help='epochs to train only the full-channel subnet')

    return parser.parse_args()
