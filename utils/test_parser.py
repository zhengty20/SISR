import argparse

def test_parser():
    parser = argparse.ArgumentParser(description='FSRCNN Testing')
    
    parser.add_argument('--model_name', type=str, default='FSRCNN', help='模型名称')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4], help='超分倍数')
    parser.add_argument('--channel_nums', type=int, default=36, help='通道数')
    parser.add_argument('--num_blocks', type=int, default=5, help='ECB块数')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--in_channels', type=int, default=1, choices=[1, 3], help='输入通道数，1表示Y通道，3表示RGB')
    
    return parser.parse_args()
