import argparse


def arm_test_parser():
    parser = argparse.ArgumentParser(description="ARMSR Testing")
    parser.add_argument("--branch_model", type=str, default="QDPSR", choices=["DPSR", "QDPSR"], help="残差分支模型")
    parser.add_argument("--checkpoint", type=str, default="", help="残差分支权重路径，留空则使用默认路径")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4], help="超分倍数")
    parser.add_argument("--channel_nums", type=int, default=32, help="通道数")
    parser.add_argument("--num_blocks", type=int, default=6, help="ECB块数")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument("--in_channels", type=int, default=3, choices=[1, 3], help="输入通道数")
    parser.add_argument("--w_bits", type=int, default=4, help="权重量化位数，仅QDPSR有效")
    parser.add_argument("--a_bits", type=int, default=8, help="激活量化位数，仅QDPSR有效")
    parser.add_argument("--arm_patch_size", type=int, default=16, help="ARMSR低分辨率分块边长")
    parser.add_argument("--arm_overlap", type=int, default=4, help="ARMSR低分辨率分块重叠像素")
    parser.add_argument("--arm_threshold", type=float, default=20.0, help="Laplace分流阈值")
    parser.add_argument("--arm_subnet_channels", type=str, default="0,16,32", help="shared_subnet 通道配置，逗号分隔")
    parser.add_argument("--arm_subnet_thresholds", type=str, default="", help="分流阈值，数量应为通道数减1，留空则按arm_threshold自动生成")
    parser.add_argument("--val_root", type=str, default="/home/tyzheng/Datasets_pt/val", help="验证集根目录")
    parser.add_argument("--clip_ratio", type=float, default=1.0, help="指标裁剪比例")
    return parser.parse_args()
