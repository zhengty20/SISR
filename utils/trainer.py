import torch
from tqdm import tqdm
from utils import metrics
from models import bilinear_interpolation
import torch.nn.functional as F

def train_epoch(model, train_loader, loss_func, optimizer, device, epoch, ema=None):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False)
    for lr_img, hr_img in pbar:
       
        lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        hr_img = (hr_img - bilinear_interpolation(lr_img, model.scale, bit8=True)) / 255.
        sr_img = model(lr_img / 255.)
        loss = loss_func(sr_img, hr_img)
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update()
        
        running_loss += loss.item()

    return running_loss / len(train_loader)

def validate_epoch(model, val_loader, loss_func, device):
    """验证一个epoch，随机采样指定数量的图片"""
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        vpbar = tqdm(val_loader, desc='loss-validating', leave=False)
        for lr_img, hr_img in vpbar:
                
            lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
            hr_img = (hr_img - bilinear_interpolation(lr_img, model.scale, bit8=True)) / 255.
            sr_img = model(lr_img / 255.) 
            loss = loss_func(sr_img, hr_img)
            val_loss += loss.item()
            
    return val_loss / len(val_loader)

def validate_metrics(model, val_loader, scale, device, clip_ratio=1.0):
    
    model.eval()
    # 存储(psnr, ssim)对
    metrics_list = []

    with torch.no_grad():
        vpbar = tqdm(val_loader, desc='metric-validating', leave=False)
        for lr_img, hr_img in vpbar:
                
            lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
            sr_img = (model(lr_img / 255.) * 255. + bilinear_interpolation(lr_img, model.scale, bit8=True)).round().clamp(0, 255)
            
            crop_border = scale
            sr_img = sr_img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            hr_img = hr_img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            
            psnr = metrics.calculate_psnr(sr_img.squeeze(0), hr_img.squeeze(0))
            ssim = metrics.calculate_ssim(sr_img.squeeze(0), hr_img.squeeze(0))
            
            metrics_list.append((psnr, ssim))
    
    if clip_ratio < 1.0:
        metrics_list.sort(key=lambda x: x[0], reverse=True)
        selected_count = max(1, int(len(metrics_list) * clip_ratio))
        selected_metrics = metrics_list[:selected_count]
    else:
        selected_metrics = metrics_list
    
    # 分别计算选中样本的psnr和ssim平均值
    psnr_list = [item[0] for item in selected_metrics]
    ssim_list = [item[1] for item in selected_metrics]

    return {
        'psnr': sum(psnr_list) / len(psnr_list),
        'ssim': sum(ssim_list) / len(ssim_list)
    }

def basic_metrics(val_loader, scale, device):
    
    # 存储(psnr, ssim)对
    metrics_list = []

    with torch.no_grad():
        vpbar = tqdm(val_loader, desc='basic-metrics-validating', leave=False)
        for lr_img, hr_img in vpbar:
                
            lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
            sr_img = F.interpolate(lr_img, scale_factor=scale, mode='bicubic', align_corners=False)

            crop_border = scale
            sr_img = sr_img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            hr_img = hr_img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            
            psnr = metrics.calculate_psnr(sr_img.squeeze(0), hr_img.squeeze(0))
            ssim = metrics.calculate_ssim(sr_img.squeeze(0), hr_img.squeeze(0))
            
            metrics_list.append((psnr, ssim))
    
    # 分别计算psnr和ssim平均值
    psnr_list = [item[0] for item in metrics_list]
    ssim_list = [item[1] for item in metrics_list]

    return {
        'psnr': sum(psnr_list) / len(psnr_list),
        'ssim': sum(ssim_list) / len(ssim_list)
    }

def transfer_weights(model, qmodel):
    # 检查模型结构是否兼容
    if len(model.body) != len(qmodel.body):
        raise ValueError(f"模型body层数不匹配: DPSR({len(model.body)}) vs QDPSR({len(qmodel.body)})")

    def _copy_conv(src_conv, dst_qconv):
        dst_qconv.conv.weight.data.copy_(src_conv.weight.data)
        if src_conv.bias is not None and dst_qconv.conv.bias is not None:
            dst_qconv.conv.bias.data.copy_(src_conv.bias.data)

    with torch.no_grad():
        # 1. 迁移head层权重
        print("迁移head层权重...")
        _copy_conv(model.head, qmodel.head)

        # 2. 迁移body层权重
        print("迁移body层权重...")
        for i in range(len(model.body)):
            print(f"  迁移第{i + 1}个Block...")
            _copy_conv(model.body[i].projection1, qmodel.body[i].projection1)
            _copy_conv(model.body[i].filter1, qmodel.body[i].filter1)
            _copy_conv(model.body[i].projection2, qmodel.body[i].projection2)
            _copy_conv(model.body[i].filter2, qmodel.body[i].filter2)

            # 迁移PReLU参数
            qmodel.body[i].act1.weight.data.copy_(model.body[i].act1.weight.data)
            qmodel.body[i].act2.weight.data.copy_(model.body[i].act2.weight.data)

        # 3. 迁移tail层权重
        print("迁移tail层权重...")
        _copy_conv(model.tail, qmodel.tail)

        # 4. 迁移alpha参数
        print("迁移alpha参数...")
        qmodel.alpha.data.copy_(model.alpha.data)

    print("权重迁移完成!")