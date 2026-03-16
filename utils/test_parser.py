import argparse

def test_parser():
    parser = argparse.ArgumentParser(description='ISCSR Testing')
    
    parser.add_argument('--model_name', type=str, default='ISCSR', help='模型名称')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4], help='超分倍数')
    parser.add_argument('--channel_nums', type=int, default=64, help='通道数')
    parser.add_argument('--num_blocks', type=int, default=4, help='ECB块数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    
    return parser.parse_args()
