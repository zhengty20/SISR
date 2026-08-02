import argparse


def test_parser():
    parser = argparse.ArgumentParser(description='Testing')

    parser.add_argument('--model_name', type=str, default='DPSR', help='model name')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/DPSR_x2_0801_1611.pth', help='DPSR checkpoint')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4], help='super-resolution scale')
    parser.add_argument('--channel_nums', type=int, default=32, help='feature channel count')
    parser.add_argument('--num_blocks', type=int, default=5, help='number of blocks')
    parser.add_argument('--block_nums', nargs='+', type=int, default=[3, 5], help='active DPSR block depths to evaluate, e.g. --block_nums 3 5')
    parser.add_argument('--subnet_expand_block', type=int, default=3, help='1-based subnet expansion block')
    parser.add_argument('--device', type=str, default='cuda', help='test device')
    parser.add_argument('--in_channels', type=int, default=3, choices=[1, 3], help='input channels, 1 for Y and 3 for RGB')
    parser.add_argument('--val_root', type=str, default='/home/tyzheng/Datasets_pt/val', help='validation set root')
    parser.add_argument('--datasets', nargs='+', default=['Set5'], help='validation datasets to test')
    parser.add_argument(
        '--width_mults',
        nargs='+',
        type=float,
        choices=[1.0, 0.5],
        default=[1.0, 0.5],
        help='DPSR widths to compare',
    )
    parser.add_argument(
        '--collect_distributions',
        action='store_true',
        help='collect and plot layer input distributions',
    )

    return parser.parse_args()
