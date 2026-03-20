import torch
from models import ARMSR
from utils import create_val_loader, metrics

def val_metrics(val_loader, scale, device, clip_ratio=0.8):
    
    frame = ARMSR(patch_size=(16, 16), scale_factor=2, overlap=4, device=device)
    
    metrics_list = []
    
    for lr_img, hr_img in val_loader:
                
        lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
        sr_img = frame.full_pipeline(lr_img.squeeze(0))
        
        crop_border = scale
        sr_img = sr_img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        hr_img = hr_img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        
        psnr = metrics.calculate_psnr(sr_img.squeeze(0), hr_img.squeeze(0))
        ssim = metrics.calculate_ssim(sr_img.squeeze(0), hr_img.squeeze(0))
        
        metrics_list.append((psnr, ssim))

    metrics_list.sort(key=lambda x: x[0], reverse=True)

    selected_count = int(len(metrics_list) * clip_ratio)
    selected_metrics = metrics_list[:selected_count]
    
    psnr_list = [item[0] for item in selected_metrics]
    ssim_list = [item[1] for item in selected_metrics]

    return {
        'psnr': sum(psnr_list) / len(psnr_list),
        'ssim': sum(ssim_list) / len(ssim_list)
    }
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_loader1 = create_val_loader('/home/tyzheng/Datasets_pt/val/Set5')
    val_loader2 = create_val_loader('/home/tyzheng/Datasets_pt/val/Set14')
    val_loader3 = create_val_loader('/home/tyzheng/Datasets_pt/val/B100')
    val_loader4 = create_val_loader('/home/tyzheng/Datasets_pt/val/U100')
    val_loader5 = create_val_loader('/home/tyzheng/Datasets_pt/val/M109')

    result_set5 = val_metrics(val_loader1, 2, device, clip_ratio=1)
    print(f'Set5: PSNR: {result_set5["psnr"]:.2f}, SSIM: {result_set5["ssim"]:.4f}')
    
    result_set14 = val_metrics(val_loader2, 2, device, clip_ratio=1)
    print(f'Set14: PSNR: {result_set14["psnr"]:.2f}, SSIM: {result_set14["ssim"]:.4f}')

    result_b100 = val_metrics(val_loader3, 2, device, clip_ratio=1)
    print(f'B100: PSNR: {result_b100["psnr"]:.2f}, SSIM: {result_b100["ssim"]:.4f}')
    
    result_u100 = val_metrics(val_loader4, 2, device, clip_ratio=1)
    print(f'U100: PSNR: {result_u100["psnr"]:.2f}, SSIM: {result_u100["ssim"]:.4f}')
    
    result_m109 = val_metrics(val_loader5, 2, device, clip_ratio=1)
    print(f'M109: PSNR: {result_m109["psnr"]:.2f}, SSIM: {result_m109["ssim"]:.4f}') 
    
if __name__ == "__main__":
    main()