import argparse

def train_parser():
    
    parser = argparse.ArgumentParser(description='Training')
    
    parser.add_argument('--model_name', type=str, default='DPSR', help='模型名称')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4], help='超分倍数')
    parser.add_argument('--channel_nums', type=int, default=32, help='通道数')
    parser.add_argument('--num_blocks', type=int, default=5, help='Block数')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-3, help='初始学习率')
    parser.add_argument('--minlr', type=float, default=1e-5, help='最小学习率')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载器工作进程数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--in_channels', type=int, default=3, choices=[1, 3], help='输入通道数，1表示Y通道，3表示RGB')
    parser.add_argument('--patch_size', type=int, default=0, help='训练HR裁剪尺寸，0表示按倍率自动选择')
    parser.add_argument('--w_bits', type=int, default=4, help='权重量化位宽')
    parser.add_argument('--a_bits', type=int, default=4, help='激活量化位宽')
    parser.add_argument(
        '--quant_method',
        type=str,
        default='lsq_plus',
        choices=['rlq', 'pact_sawb', 'lsq_plus'],
        help='量化方法，选择不同QConv2d实现'
    )
    parser.add_argument('--warmup_epochs', type=int, default=15, help='warmup轮数')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA衰减系数')
    parser.add_argument('--shared_subnet_channels', type=int, default=16, help='shared子网通道数')
    parser.add_argument('--shared_full_epochs', type=int, default=50, help='前N轮仅训练全通道(C32)')
    
    return parser.parse_args()