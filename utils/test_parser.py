import argparse


def test_parser():
    parser = argparse.ArgumentParser(description='Testing')

    parser.add_argument('--model_name', type=str, default='DPSR', help='model name')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4], help='super-resolution scale')
    parser.add_argument('--channel_nums', type=int, default=36, help='feature channel count')
    parser.add_argument('--num_blocks', type=int, default=5, help='number of blocks')
    parser.add_argument('--device', type=str, default='cuda', help='test device')
    parser.add_argument('--in_channels', type=int, default=3, choices=[1, 3], help='input channels, 1 for Y and 3 for RGB')
    parser.add_argument('--shared_subnet_channels', type=int, default=18, help='shared subnet channel count')

    return parser.parse_args()
