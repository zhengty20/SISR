import argparse

def test_parser():
    parser = argparse.ArgumentParser(description='Testing')
    
    parser.add_argument('--model_name', type=str, default='DPSR', help='模型名称')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4], help='超分倍数')
    parser.add_argument('--channel_nums', type=int, default=36, help='通道数')
    parser.add_argument('--num_blocks', type=int, default=5, help='ECB块数')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--in_channels', type=int, default=3, choices=[1, 3], help='输入通道数，1表示Y通道，3表示RGB')
    parser.add_argument('--shared_subnet_channels', type=int, default=18, help='共享子网通道数')
    parser.add_argument('--w_bits', type=int, default=4, help='权重量化位数')
    parser.add_argument('--a_bits', type=int, default=4, help='激活量化位数')
    parser.add_argument(
        '--quant_method',
        type=str,
        default='lsq_plus',
        choices=['rlq', 'pact_sawb', 'lsq_plus'],
        help='量化方法，选择不同QConv2d实现'
    )
    
    return parser.parse_args()
