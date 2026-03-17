import torch
from tqdm import tqdm
from utils import metrics
from models import usm_interpolation
import torch.nn.functional as F

def train_epoch(model, train_loader, augmentor, loss_func, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False)
    for hr_img in pbar:
        
        hr_img = hr_img.to(device).float()
        lr_img, hr_img = augmentor(hr_img)
        hr_img = (hr_img - usm_interpolation(lr_img, model.scale, bit8=True)).div(255.)
        
        lr_img = lr_img.div(255.)
        sr_img = model(lr_img)

        loss = loss_func(sr_img, hr_img)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
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
            hr_img = (hr_img - usm_interpolation(lr_img, model.scale, bit8=True)).div(255.)
            
            lr_img = lr_img.div(255.)
            sr_img = model(lr_img)           
            
            loss = loss_func(sr_img, hr_img)
            val_loss += loss.item()
            
    return val_loss / len(val_loader)

def validate_metrics(model, val_loader, scale, device, clip_ratio=0.8):
    
    model.eval()
    # 存储(psnr, ssim)对
    metrics_list = []

    with torch.no_grad():
        vpbar = tqdm(val_loader, desc='metric-validating', leave=False)
        for lr_img, hr_img in vpbar:
                
            lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
            
            lr_img_norm = lr_img.div(255.)
            sr_img_norm = model(lr_img_norm)
            sr_img = ((sr_img_norm * 255.).round() + usm_interpolation(lr_img, model.scale, bit8=True)).clamp(0, 255)
            
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

def basic_metrics(val_loader, scale, device):
    
    # 存储(psnr, ssim)对
    metrics_list = []

    vpbar = tqdm(val_loader, desc='basic-metrics-validating', leave=False)
    for lr_img, hr_img in vpbar:
            
        lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
        sr_img = F.interpolate(lr_img, scale_factor=scale, mode='bicubic', align_corners=False)
        
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
        raise ValueError(f"模型body层数不匹配: ISCSR({len(model.body)}) vs QISCSR({len(qmodel.body)})")
    
    # 1. 迁移head层权重
    print("迁移head层权重...")
    qmodel.head.conv.weight.data.copy_(model.head.weight.data)
    if model.head.bias is not None and qmodel.head.conv.bias is not None:
        qmodel.head.conv.bias.data.copy_(model.head.bias.data)
    
    # 2. 迁移body层权重
    print("迁移body层权重...")
    for i in range(len(model.body)):
        print(f"  迁移第{i+1}个Block...")
        
        # 迁移filter1
        qmodel.body[i].filter1.conv.weight.data.copy_(model.body[i].filter1.weight.data)
        if model.body[i].filter1.bias is not None and qmodel.body[i].filter1.conv.bias is not None:
            qmodel.body[i].filter1.conv.bias.data.copy_(model.body[i].filter1.bias.data)
        
        # 迁移filter2
        qmodel.body[i].filter2.conv.weight.data.copy_(model.body[i].filter2.weight.data)
        if model.body[i].filter2.bias is not None and qmodel.body[i].filter2.conv.bias is not None:
            qmodel.body[i].filter2.conv.bias.data.copy_(model.body[i].filter2.bias.data)
        
        # 迁移projection1
        qmodel.body[i].projection1.conv.weight.data.copy_(model.body[i].projection1.weight.data)
        if model.body[i].projection1.bias is not None and qmodel.body[i].projection1.conv.bias is not None:
            qmodel.body[i].projection1.conv.bias.data.copy_(model.body[i].projection1.bias.data)
        
        # 迁移projection2
        qmodel.body[i].projection2.conv.weight.data.copy_(model.body[i].projection2.weight.data)
        if model.body[i].projection2.bias is not None and qmodel.body[i].projection2.conv.bias is not None:
            qmodel.body[i].projection2.conv.bias.data.copy_(model.body[i].projection2.bias.data)
    
    # 3. 迁移tail层权重
    print("迁移tail层权重...")
    qmodel.tail.conv.weight.data.copy_(model.tail.weight.data)
    if model.tail.bias is not None and qmodel.tail.conv.bias is not None:
        qmodel.tail.conv.bias.data.copy_(model.tail.bias.data)
    
    # 4. 迁移alpha参数
    print("迁移alpha参数...")
    qmodel.alpha.data.copy_(model.alpha.data)
    
    print("权重迁移完成!")