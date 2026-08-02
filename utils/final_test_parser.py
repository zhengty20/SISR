import argparse


def final_test_parser():
    parser = argparse.ArgumentParser(description="DPSR Testing")
    parser.add_argument("--checkpoint", type=str, default="", help="残差分支权重路径，留空则使用默认路径")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4], help="超分倍数")
    parser.add_argument("--channel_nums", type=int, default=32, help="通道数")
    parser.add_argument("--num_blocks", type=int, default=5, help="ECB块数")
    parser.add_argument("--subnet_expand_block", type=int, default=3, help="压缩子网恢复完整通道的Block序号（从1开始）")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument("--in_channels", type=int, default=3, choices=[1, 3], help="输入通道数")
    parser.add_argument("--w_bits", type=int, default=4, help="权重量化位数，仅QDPSR有效")
    parser.add_argument("--a_bits", type=int, default=8, help="激活量化位数，仅QDPSR有效")
    parser.add_argument("--arm_patch_size", type=int, default=24, help="ARMSR低分辨率分块边长")
    parser.add_argument("--arm_overlap", type=int, default=2, help="ARMSR低分辨率分块重叠像素")
    parser.add_argument("--arm_threshold", type=float, default=14.5, help="进入完整通道NN的Laplace高阈值")
    parser.add_argument("--arm_subnet_threshold", type=float, default=3.5, help="从bicubic进入压缩通道NN的Laplace低阈值")
    parser.add_argument("--arm_subnet_width_mult", type=float, default=0.5, choices=[0.5], help="压缩通道NN宽度倍率")
    parser.add_argument("--val_root", type=str, default="/home/tyzheng/Datasets_pt/val", help="验证集根目录")
    parser.add_argument("--clip_ratio", type=float, default=1, help="指标裁剪比例")
    return parser.parse_args()