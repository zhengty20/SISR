import os
import sys
import torch
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from torch_ema import ExponentialMovingAverage
import copy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import ISCSR, QISCSR
from utils import train_parser, train_epoch, validate_epoch, validate_metrics, transfer_weights, \
create_logger, create_train_loader, create_val_loader, SRKorniaAugmentor, MixedLoss

def main():

    args = train_parser()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 创建logger
    logger = create_logger(log_dir="./logs", model_name=args.model_name, scale=args.scale)
    logger.info(f"使用设备: {device}")

    train_loader = create_train_loader('/home/tyzheng/Datasets_pt/train', batch_size=args.batch_size, num_workers=args.num_workers)
                                   
    val_loader_set5 = create_val_loader('/home/tyzheng/Datasets_pt/val/Set5', args.scale)
    val_loader_set14 = create_val_loader('/home/tyzheng/Datasets_pt/val/Set14', args.scale)
    val_loader_b100 = create_val_loader('/home/tyzheng/Datasets_pt/val/B100', args.scale)
    val_loader_u100 = create_val_loader('/home/tyzheng/Datasets_pt/val/U100', args.scale)
    val_loader_m109 = create_val_loader('/home/tyzheng/Datasets_pt/val/M109', args.scale)
    
    time_stamp = datetime.now().strftime("%m%d_%H%M")
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{args.model_name}_x{args.scale}_{time_stamp}.pth"
    model = QISCSR(scale = args.scale, in_dim = 3, fea_dim = args.channel_nums, num_blocks = args.num_blocks, bias = False).to(device)
    # model = ISCSR(scale = args.scale, in_dim = 3, fea_dim = args.channel_nums, num_blocks = args.num_blocks, bias = False).to(device)

    # 统计模型参数量
    total_params = model.param_num()
    logger.info(f"模型总参数量: {total_params:,}")

    augmentor = SRKorniaAugmentor(scale=args.scale)
    augmentor = augmentor.to(device)
    
    # 损失函数  
    loss_func = MixedLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr)
    
    # 排除量化器的scale参数，只对卷积权重和偏置使用EMA
    ema_params = []
    for name, param in model.named_parameters():
        if 'quantizer.s' not in name:  # 排除量化器的scale参数
            ema_params.append(param)
    
    ema = ExponentialMovingAverage(ema_params, decay=0.995)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.minlr)
   
    # 记录训练开始信息
    logger.log_training_start(args, total_params, len(train_loader), 
                              len(val_loader_set5) + len(val_loader_set14) + len(val_loader_b100) + len(val_loader_u100) + len(val_loader_m109))


    # 训练循环
    best_val_loss = 10

    logger.info("Begin Training")
    for epoch in range(args.epochs):

        # 训练
        val_loss = 0.0
        train_loss = train_epoch(model, train_loader, augmentor, loss_func, optimizer, device, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.log_epoch_train(epoch, args.epochs, train_loss, current_lr)
        ema.update()
        
        with ema.average_parameters():
            
            val_loss = validate_epoch(model, val_loader_set5, loss_func, device)
            val_loss += validate_epoch(model, val_loader_set14, loss_func, device)
            val_loss += validate_epoch(model, val_loader_b100, loss_func, device)
            val_loss += validate_epoch(model, val_loader_u100, loss_func, device)
            val_loss += validate_epoch(model, val_loader_m109, loss_func, device)
            val_loss /= 5

        logger.log_epoch_val(epoch, args.epochs, val_loss)
        
        if val_loss < best_val_loss:
            
            best_val_loss = val_loss
            ema_model = copy.deepcopy(model)
            
            # 只复制EMA管理的参数（排除量化器的scale参数）
            ema_params_list = []
            for name, param in ema_model.named_parameters():
                if 'quantizer.s' not in name:
                    ema_params_list.append(param)
            
            ema.copy_to(ema_params_list)
            
            torch.save({
                'epoch': epoch + 1,
                'iteration': (epoch + 1) * len(train_loader),
                'model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, model_path)
            
            logger.log_best_model(val_loss)

            val_metrics_set5 = validate_metrics(ema_model, val_loader_set5, args.scale, device, 1)
            logger.log_validation_results("Set5", val_metrics_set5)
            
            val_metrics_set14 = validate_metrics(ema_model, val_loader_set14, args.scale, device, 1)
            logger.log_validation_results("Set14", val_metrics_set14)
            
            val_metrics_b100 = validate_metrics(ema_model, val_loader_b100, args.scale, device, 1)
            logger.log_validation_results("B100", val_metrics_b100)
            
            val_metrics_u100 = validate_metrics(ema_model, val_loader_u100, args.scale, device, 1)
            logger.log_validation_results("U100", val_metrics_u100)
            
            val_metrics_m109 = validate_metrics(ema_model, val_loader_m109, args.scale, device, 1)
            logger.log_validation_results("M109", val_metrics_m109)

        # 更新学习率（在epoch结束时）
        scheduler.step()
    
    logger.log_training_finished()

    logger.log_testing_start("Best Model")
    net = QISCSR(scale = args.scale, in_dim = 3, fea_dim = args.channel_nums, num_blocks = args.num_blocks, bias = False).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        _ = net(dummy_input)
    
    net.load_state_dict(state_dict['model_state_dict'])
    net.eval()

    val_metrics = validate_metrics(net, val_loader_set5, args.scale, device, 1)
    logger.log_validation_results("Set5", val_metrics)
    
    val_metrics = validate_metrics(net, val_loader_set14, args.scale, device, 1)
    logger.log_validation_results("Set14", val_metrics)
    
    val_metrics = validate_metrics(net, val_loader_b100, args.scale, device, 1)
    logger.log_validation_results("B100", val_metrics)
    
    val_metrics = validate_metrics(net, val_loader_u100, args.scale, device, 1)
    logger.log_validation_results("U100", val_metrics)
    
    val_metrics = validate_metrics(net, val_loader_m109, args.scale, device, 1)
    logger.log_validation_results("M109", val_metrics)

    # 关闭logger
    logger.close()

if __name__ == "__main__":
    main()