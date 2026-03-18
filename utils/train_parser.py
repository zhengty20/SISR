import argparse

def train_parser():
    
    parser = argparse.ArgumentParser(description='DPSR Training')
    
    parser.add_argument('--model_name', type=str, default='DPSR', help='模型名称')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4], help='超分倍数')
    parser.add_argument('--channel_nums', type=int, default=36, help='通道数')
    parser.add_argument('--num_blocks', type=int, default=5, help='ECB块数')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--minlr', type=float, default=1e-5, help='最小学习率')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载器工作进程数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--patch_size', type=int, default=0, help='训练HR裁剪尺寸，0表示按倍率自动选择')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup轮数')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA衰减系数')
    parser.add_argument('--retrain', type=bool, default=False, help='是否继续训练')
    
    return parser.parse_args()