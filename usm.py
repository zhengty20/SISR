
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import create_val_loader, metrics
from models import usm_interpolation

scale = 2
device = 'cuda'

def validate_metrics(val_loader, scale, device):
    
    metrics_list = []  # 存储(psnr, ssim)对

    with torch.no_grad():
        vpbar = tqdm(val_loader, desc='metric-validating', leave=False)
        for lr_img, hr_img in vpbar:
                
            lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
            sr_img = usm_interpolation(lr_img, scale)
            
            crop_border = scale
            sr_img = sr_img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            hr_img = hr_img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            
            psnr = metrics.calculate_psnr(sr_img.squeeze(0), hr_img.squeeze(0))
            ssim = metrics.calculate_ssim(sr_img.squeeze(0), hr_img.squeeze(0))
            
            metrics_list.append((psnr, ssim))
    
    # 分别计算选中样本的psnr和ssim平均值
    psnr_list = [item[0] for item in metrics_list]
    ssim_list = [item[1] for item in metrics_list]

    return {
        'psnr': sum(psnr_list) / len(psnr_list),
        'ssim': sum(ssim_list) / len(ssim_list)
    }

if __name__ == '__main__':

    scale = 2
    
    val_loader_set5 = create_val_loader('/home/tyzheng/Datasets_pt/val/Set5', scale)
    val_loader_set14 = create_val_loader('/home/tyzheng/Datasets_pt/val/Set14', scale)
    val_loader_b100 = create_val_loader('/home/tyzheng/Datasets_pt/val/B100', scale)
    val_loader_u100 = create_val_loader('/home/tyzheng/Datasets_pt/val/U100', scale)
    val_loader_m109 = create_val_loader('/home/tyzheng/Datasets_pt/val/M109', scale)

    result_set5 = validate_metrics(val_loader_set5, scale, device)
    print(f'Set5: PSNR: {result_set5["psnr"]:.2f}, SSIM: {result_set5["ssim"]:.4f}')

    result_set14 = validate_metrics(val_loader_set14, scale, device)
    print(f'Set14: PSNR: {result_set14["psnr"]:.2f}, SSIM: {result_set14["ssim"]:.4f}')

    result_b100 = validate_metrics(val_loader_b100, scale, device)
    print(f'B100: PSNR: {result_b100["psnr"]:.2f}, SSIM: {result_b100["ssim"]:.4f}')

    result_u100 = validate_metrics(val_loader_u100, scale, device)
    print(f'U100: PSNR: {result_u100["psnr"]:.2f}, SSIM: {result_u100["ssim"]:.4f}')

    result_div2k = validate_metrics(val_loader_m109, scale, device)
    print(f'M109: PSNR: {result_div2k["psnr"]:.2f}, SSIM: {result_div2k["ssim"]:.4f}') 
