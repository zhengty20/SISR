import torch

from utils import create_val_loader, test_parser, validate_metrics
from models import QDPSR

if __name__ == '__main__':

    args = test_parser()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    net = QDPSR(scale=args.scale, in_dim=args.in_channels, fea_dim=args.channel_nums, num_blocks=args.num_blocks, bias=False, weight_bitwidth=args.w_bits, activation_bitwidth=args.a_bits).to(device)
    checkpoint = torch.load("./checkpoints/QDPSR_x2_0323_1639.pth", map_location=device, weights_only=False)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    net.load_state_dict(model_state_dict)
    net.eval()

    val_loaders = {
        'Set5': create_val_loader('/home/tyzheng/Datasets_pt/val/Set5', args.scale, in_channels=args.in_channels),
        'Set14': create_val_loader('/home/tyzheng/Datasets_pt/val/Set14', args.scale, in_channels=args.in_channels),
        'B100': create_val_loader('/home/tyzheng/Datasets_pt/val/B100', args.scale, in_channels=args.in_channels),
        'U100': create_val_loader('/home/tyzheng/Datasets_pt/val/U100', args.scale, in_channels=args.in_channels),
        'M109': create_val_loader('/home/tyzheng/Datasets_pt/val/M109', args.scale, in_channels=args.in_channels),
    }

    for dataset_name, loader in val_loaders.items():
        result = validate_metrics(net, loader, args.scale, device, 1.0)
        print(f'{dataset_name}: PSNR: {result["psnr"]:.2f}, SSIM: {result["ssim"]:.4f}')