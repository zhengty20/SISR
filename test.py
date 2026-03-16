import torch
from tqdm import tqdm

from utils import create_val_loader, metrics, test_parser
from models import usm_interpolation, DPSR

def validate_metrics(model, val_loader, scale, device, clip_ratio=0.8):
    
    metrics_list = []  # 存储(psnr, ssim)对

    with torch.no_grad():
        vpbar = tqdm(val_loader, desc='metric-validating', leave=False)
        for lr_img, hr_img in vpbar:
                
            lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
            sr_img = usm_interpolation(lr_img, scale)

            lr_img_norm = lr_img.div(255.)
            sr_img_norm = model(lr_img_norm)
            sr_img = ((sr_img_norm * 255.).round() + usm_interpolation(lr_img, model.scale)).clamp(0, 255)
            
            crop_border = scale
            sr_img = sr_img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            hr_img = hr_img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            
            psnr = metrics.calculate_psnr(sr_img.squeeze(0), hr_img.squeeze(0))
            ssim = metrics.calculate_ssim(sr_img.squeeze(0), hr_img.squeeze(0))
            
            metrics_list.append((psnr, ssim))
    
    # 按照psnr值排序（降序）
    metrics_list.sort(key=lambda x: x[0], reverse=True)
    
    # 选择前clip_ratio比例的样本
    selected_count = int(len(metrics_list) * clip_ratio)
    selected_metrics = metrics_list[:selected_count]
    
    # 分别计算选中样本的psnr和ssim平均值
    psnr_list = [item[0] for item in selected_metrics]
    ssim_list = [item[1] for item in selected_metrics]

    return {
        'psnr': sum(psnr_list) / len(psnr_list),
        'ssim': sum(ssim_list) / len(ssim_list)
    }

if __name__ == '__main__':

    args = test_parser()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    net = DPSR(scale = args.scale, in_dim = 3, fea_dim = args.channel_nums, num_blocks = args.num_blocks, bias = False).to(device)
    state_dict = torch.load("./checkpoints/ISCSR_x2_0920_1337.pth", map_location=device, weights_only=False)
    
    net.load_state_dict(state_dict['model_state_dict'])
    net.eval()
    
    val_loader_set5 = create_val_loader('/home/tyzheng/Datasets_pt/val/Set5', args.scale)
    val_loader_set14 = create_val_loader('/home/tyzheng/Datasets_pt/val/Set14', args.scale)
    val_loader_b100 = create_val_loader('/home/tyzheng/Datasets_pt/val/B100', args.scale)
    val_loader_u100 = create_val_loader('/home/tyzheng/Datasets_pt/val/U100', args.scale)
    val_loader_m109 = create_val_loader('/home/tyzheng/Datasets_pt/val/M109', args.scale)

    result_set5 = validate_metrics(net, val_loader_set5, args.scale, device, 0.8)
    print(f'Set5: PSNR: {result_set5["psnr"]:.2f}, SSIM: {result_set5["ssim"]:.4f}')

    result_set14 = validate_metrics(net, val_loader_set14, args.scale, device, 0.8)
    print(f'Set14: PSNR: {result_set14["psnr"]:.2f}, SSIM: {result_set14["ssim"]:.4f}')

    result_b100 = validate_metrics(net, val_loader_b100, args.scale, device, 0.8)
    print(f'B100: PSNR: {result_b100["psnr"]:.2f}, SSIM: {result_b100["ssim"]:.4f}')

    result_u100 = validate_metrics(net, val_loader_u100, args.scale, device, 0.8)
    print(f'U100: PSNR: {result_u100["psnr"]:.2f}, SSIM: {result_u100["ssim"]:.4f}')

    result_div2k = validate_metrics(net, val_loader_m109, args.scale, device, 0.8)
    print(f'M109: PSNR: {result_div2k["psnr"]:.2f}, SSIM: {result_div2k["ssim"]:.4f}') 