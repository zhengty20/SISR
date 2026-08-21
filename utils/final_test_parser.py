import argparse


def final_test_parser():
    parser = argparse.ArgumentParser(description="Dynamic DPSR and FSRCNN testing")
    parser.add_argument(
        "--dpsr-checkpoint",
        "--dpsr_checkpoint",
        "--checkpoint",
        dest="dpsr_checkpoint",
        type=str,
        required=True,
        help="dynamic DPSR checkpoint path; --checkpoint is a compatibility alias",
    )
    parser.add_argument(
        "--fsrcnn-checkpoint",
        "--fsrcnn_checkpoint",
        dest="fsrcnn_checkpoint",
        type=str,
        required=True,
        help="FSRCNN checkpoint path",
    )
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4], help="超分倍数")
    parser.add_argument("--channel_nums", type=int, default=32, choices=[32], help="完整路径通道数")
    parser.add_argument("--subnet_channels", type=int, default=16, help="子网显式通道数")
    parser.add_argument("--num_blocks", type=int, default=5, help="DPSR网络的Block数")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument("--in_channels", type=int, default=3, choices=[1, 3], help="输入通道数")
    parser.add_argument("--arm_patch_size", type=int, default=24, help="ARMSR低分辨率分块边长")
    parser.add_argument("--arm_overlap", type=int, default=2, help="ARMSR低分辨率分块重叠像素")
    parser.add_argument("--arm_threshold", type=float, default=12.5, help="进入完整通道NN的Laplace高阈值")
    parser.add_argument("--arm_subnet_threshold", type=float, default=2.5, help="从bilinear进入子网NN的Laplace阈值")
    parser.add_argument("--val_root", type=str, default="/home/tyzheng/Datasets_pt/val", help="验证集根目录")
    parser.add_argument("--clip_ratio", type=float, default=1, help="指标裁剪比例")
    return parser.parse_args()
