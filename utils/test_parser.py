import argparse

def test_parser():
    parser = argparse.ArgumentParser(description='DPSR Testing')
    
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4], help='超分倍数')
    parser.add_argument('--channel_nums', type=int, default=36, help='通道数')
    parser.add_argument('--num_blocks', type=int, default=5, help='ECB块数')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--checkpoint', type=str, required=True, help='待测试的模型权重路径')
    
    return parser.parse_args()
