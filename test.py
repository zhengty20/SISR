import torch

from utils import create_val_loader, test_parser, validate_metrics
from models import DPSR

if __name__ == '__main__':

    args = test_parser()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    net = DPSR(scale = args.scale, in_dim = 3, fea_dim = args.channel_nums, num_blocks = args.num_blocks, bias = False).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    net.load_state_dict(state_dict['model_state_dict'])
    net.eval()
    
    val_loader_set5 = create_val_loader('/home/tyzheng/Datasets_pt/val/Set5', args.scale)
    val_loader_set14 = create_val_loader('/home/tyzheng/Datasets_pt/val/Set14', args.scale)
    val_loader_b100 = create_val_loader('/home/tyzheng/Datasets_pt/val/B100', args.scale)
    val_loader_u100 = create_val_loader('/home/tyzheng/Datasets_pt/val/U100', args.scale)
    val_loader_m109 = create_val_loader('/home/tyzheng/Datasets_pt/val/M109', args.scale)

    result_set5 = validate_metrics(net, val_loader_set5, args.scale, device, 1.0)
    print(f'Set5: PSNR: {result_set5["psnr"]:.2f}, SSIM: {result_set5["ssim"]:.4f}')

    result_set14 = validate_metrics(net, val_loader_set14, args.scale, device, 1.0)
    print(f'Set14: PSNR: {result_set14["psnr"]:.2f}, SSIM: {result_set14["ssim"]:.4f}')

    result_b100 = validate_metrics(net, val_loader_b100, args.scale, device, 1.0)
    print(f'B100: PSNR: {result_b100["psnr"]:.2f}, SSIM: {result_b100["ssim"]:.4f}')

    result_u100 = validate_metrics(net, val_loader_u100, args.scale, device, 1.0)
    print(f'U100: PSNR: {result_u100["psnr"]:.2f}, SSIM: {result_u100["ssim"]:.4f}')

    result_div2k = validate_metrics(net, val_loader_m109, args.scale, device, 1.0)
    print(f'M109: PSNR: {result_div2k["psnr"]:.2f}, SSIM: {result_div2k["ssim"]:.4f}') 