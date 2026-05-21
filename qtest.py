import torch

from utils import create_val_loader, test_parser, validate_metrics, validate_metrics_shared_channel, bicubic_metrics, bilinear_metrics
from models import build_qdpsr


def _build_val_loaders(scale, in_channels):
    return {
        'Set5': create_val_loader('/home/tyzheng/Datasets_pt/val/Set5', scale, in_channels=in_channels),
        'Set14': create_val_loader('/home/tyzheng/Datasets_pt/val/Set14', scale, in_channels=in_channels),
        'B100': create_val_loader('/home/tyzheng/Datasets_pt/val/B100', scale, in_channels=in_channels),
        'U100': create_val_loader('/home/tyzheng/Datasets_pt/val/U100', scale, in_channels=in_channels),
        'M109': create_val_loader('/home/tyzheng/Datasets_pt/val/M109', scale, in_channels=in_channels),
    }


def _print_metrics(loaders, metric_fn, name_suffix=''):
    for dataset_name, loader in loaders.items():
        result = metric_fn(loader)
        label = f'{dataset_name}{name_suffix}'
        print(f'{label}: PSNR: {result["psnr"]:.2f}, SSIM: {result["ssim"]:.4f}')


if __name__ == '__main__':

    args = test_parser()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    net = build_qdpsr(
        scale=args.scale,
        in_dim=args.in_channels,
        fea_dim=args.channel_nums,
        num_blocks=args.num_blocks,
        bias=False,
        weight_bitwidth=args.w_bits,
        activation_bitwidth=args.a_bits,
        quant_method=args.quant_method,
    ).to(device)
    net.head.set_input_quantization(False)
    checkpoint = torch.load("./checkpoints/QDPSR_x2_0327_1659.pth", map_location=device, weights_only=False)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    net.load_state_dict(model_state_dict)
    net.eval()

    val_loaders = _build_val_loaders(args.scale, args.in_channels)

    _print_metrics(
        loaders=val_loaders,
        metric_fn=lambda loader: validate_metrics(net, loader, args.scale, device, 1.0),
    )
    _print_metrics(
        loaders=val_loaders,
        metric_fn=lambda loader: validate_metrics_shared_channel(net, loader, args.scale, device, args.shared_subnet_channels, 1.0),
        name_suffix=f'-C{args.shared_subnet_channels}',
    )

    _print_metrics(
        loaders=val_loaders,
        metric_fn=lambda loader: bicubic_metrics(loader, args.scale, device),
        name_suffix='-bicubic',
    )
    _print_metrics(
        loaders=val_loaders,
        metric_fn=lambda loader: bilinear_metrics(loader, args.scale, device),
        name_suffix='-bilinear',
    )