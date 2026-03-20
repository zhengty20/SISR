import os
import sys
import torch
import torch.optim as optim
import copy

from datetime import datetime
from pathlib import Path
from torch_ema import ExponentialMovingAverage
from models import DPSR, FSRCNN
from utils import train_parser, train_epoch, validate_epoch, validate_metrics, basic_metrics, create_logger, \
create_train_loader, create_val_loader, WarmupCosineScheduler, MixedLoss

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():

    args = train_parser()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 创建logger
    logger = create_logger(log_dir="./logs", model_name=args.model_name, scale=args.scale)
    logger.info(f"使用设备: {device}")

    train_loader = create_train_loader(
        '/home/tyzheng/Datasets_pt/train',
        scale=args.scale,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        in_channels=args.in_channels
    )
                                   
    val_loader_set5 = create_val_loader('/home/tyzheng/Datasets_pt/val/Set5', args.scale, in_channels=args.in_channels)
    val_loader_set14 = create_val_loader('/home/tyzheng/Datasets_pt/val/Set14', args.scale, in_channels=args.in_channels)
    val_loader_b100 = create_val_loader('/home/tyzheng/Datasets_pt/val/B100', args.scale, in_channels=args.in_channels)
    val_loader_u100 = create_val_loader('/home/tyzheng/Datasets_pt/val/U100', args.scale, in_channels=args.in_channels)
    val_loader_m109 = create_val_loader('/home/tyzheng/Datasets_pt/val/M109', args.scale, in_channels=args.in_channels)
    val_loaders = {
        'Set5': val_loader_set5,
        'Set14': val_loader_set14,
        'B100': val_loader_b100,
        'U100': val_loader_u100,
        'M109': val_loader_m109,
    }
    
    time_stamp = datetime.now().strftime("%m%d_%H%M")
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{args.model_name}_x{args.scale}_{time_stamp}.pth"
    model = DPSR(scale=args.scale, in_dim=args.in_channels, fea_dim=args.channel_nums, num_blocks=args.num_blocks, bias=False).to(device)
    # model = FSRCNN(scale=args.scale, in_dim=args.in_channels, d_dim=56, s_dim=12, num_blocks=4).to(device)
    
    # 统计模型参数量
    total_params = model.param_num()
    logger.info(f"模型总参数量: {total_params:,}")

    # 损失函数  
    loss_func = MixedLoss(eps=1e-8, gamma=0.2)

    # 优化器
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr)
    '''
    optimizer = optim.Adam([
        {'params': model.first_part.parameters()},
        {'params': model.mid_part.parameters()},
        {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)
    '''
    
    # EMA
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    
    # 学习率调度器：warmup + cosine annealing
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        eta_min=args.minlr,
        warmup_start_lr=2e-4
    )

    # 记录训练开始信息
    logger.log_training_start(args, total_params, len(train_loader), 
                              len(val_loader_set5) + len(val_loader_set14) + len(val_loader_b100) + len(val_loader_u100) + len(val_loader_m109))

    # 训练循环
    best_val_loss = 10.0

    logger.info("Begin Training")
    for epoch in range(args.epochs):
        # 训练
        val_loss = 0.0
        train_loss = train_epoch(model, train_loader, loss_func, optimizer, device, epoch, ema=ema)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.log_epoch_train(epoch, args.epochs, train_loss, current_lr)

        best_candidate = None
        with ema.average_parameters():
            weighted_val_loss = 0.0
            total_val_samples = 0
            for loader in val_loaders.values():
                loader_loss = validate_epoch(model, loader, loss_func, device)
                sample_count = len(loader.dataset)
                weighted_val_loss += loader_loss * sample_count
                total_val_samples += sample_count
            val_loss = weighted_val_loss / total_val_samples

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_candidate = copy.deepcopy(model)

        logger.log_epoch_val(epoch, args.epochs, val_loss)
        
        if best_candidate is not None:
            torch.save({
                'epoch': epoch + 1,
                'iteration': (epoch + 1) * len(train_loader),
                'model_state_dict': best_candidate.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, model_path)
            
            logger.log_best_model(val_loss)

            for dataset_name, loader in val_loaders.items():
                val_metrics = validate_metrics(best_candidate, loader, args.scale, device, 1.0)
                logger.log_validation_results(dataset_name, val_metrics)

        scheduler.step()
    
    logger.log_training_finished()

    logger.log_testing_start("Best Model")
    net = DPSR(scale=args.scale, in_dim=args.in_channels, fea_dim=args.channel_nums, num_blocks=args.num_blocks, bias=False).to(device)
    # net = FSRCNN(scale=args.scale, in_dim=args.in_channels, d_dim=56, s_dim=12, num_blocks=4).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    
    net.load_state_dict(state_dict['model_state_dict'])
    net.eval()

    for dataset_name, loader in val_loaders.items():
        val_metrics = validate_metrics(net, loader, args.scale, device, 1.0)
        logger.log_validation_results(dataset_name, val_metrics)

    logger.log_testing_start("Bicubic Interpolation")

    for dataset_name, loader in val_loaders.items():
        val_metrics = basic_metrics(loader, args.scale, device)
        logger.log_validation_results(dataset_name, val_metrics)

    # 关闭logger
    logger.close()

if __name__ == "__main__":
    main()