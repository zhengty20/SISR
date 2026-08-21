import os
import sys
import torch
import torch.optim as optim
import copy
from datetime import datetime
from pathlib import Path
from torch_ema import ExponentialMovingAverage

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import QConv2dLSQP, build_qdpsr
from utils import train_parser, train_epoch, validate_epoch, validate_metrics, bilinear_metrics, \
create_logger, create_train_loader, create_val_loader, WarmupCosineScheduler, MixedLoss

def _build_q_model(args, device):
    model = build_qdpsr(
        scale=args.scale,
        in_dim=args.in_channels,
        fea_dim=args.channel_nums,
        num_blocks=args.num_blocks,
        bias=False,
        weight_bitwidth=args.wbits,
        activation_bitwidth=args.abits,
        subnet_channels=args.subnet_channels,
    ).to(device)
    return model

def _load_fp_checkpoint_into_q_model(model, checkpoint_path, device, logger):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fp_state = checkpoint.get('model_state_dict', checkpoint)
    q_state = model.state_dict()
    mapped_state = {}
    block_conv_names = {'filter1', 'filter2', 'projection1', 'projection2'}

    for key, value in fp_state.items():
        q_key = key

        if key.startswith('head.'):
            q_key = 'head.conv.' + key[len('head.'):]
        elif key.startswith('tail.'):
            q_key = 'tail.conv.' + key[len('tail.'):]
        else:
            parts = key.split('.')
            if (
                len(parts) >= 4
                and parts[0] == 'body'
                and parts[2] in block_conv_names
                and parts[3] in {'weight', 'bias'}
            ):
                q_key = f'body.{parts[1]}.{parts[2]}.conv.{parts[3]}'

        if q_key in q_state and q_state[q_key].shape == value.shape:
            mapped_state[q_key] = value
        else:
            logger.info(f'Skip pretrained key: {key} -> {q_key}')

    q_state.update(mapped_state)
    model.load_state_dict(q_state)
    for module in model.modules():
        if hasattr(module, 'initialized'):
            module.initialized = False
    logger.info(f'Loaded {len(mapped_state)} tensors from full-precision checkpoint: {checkpoint_path}')

def _model_configurations(args):
    return [(f'{args.channel_nums}ch', args.channel_nums)]


def _calibrate_lsqplus_activations(model, train_loader, args, device, logger):
    """Initialize each activation quantizer from full-precision feature maps."""
    if args.quant_calibration_batches <= 0:
        return

    quantized_layers = {
        name: module for name, module in model.named_modules()
        if isinstance(module, QConv2dLSQP) and not module.act_quant.disabled
    }
    samples = {name: [] for name in quantized_layers}
    sample_counts = {name: 0 for name in quantized_layers}

    def make_hook(name):
        def hook(_module, inputs):
            feature_map = inputs[0].detach()
            channels = feature_map.shape[1]
            values = feature_map.movedim(1, 0).reshape(channels, -1)
            per_channel_limit = max(1, args.quant_calibration_samples // channels)
            remaining = per_channel_limit - sample_counts[name]
            if remaining <= 0 or values.shape[1] == 0:
                return
            take = min(remaining, max(1, 8192 // channels), values.shape[1])
            indices = torch.linspace(0, values.shape[1] - 1, take, device=values.device).long()
            samples[name].append(values[:, indices].cpu())
            sample_counts[name] += take
        return hook

    handles = [module.register_forward_pre_hook(make_hook(name)) for name, module in quantized_layers.items()]
    was_training = model.training
    model.set_quantization_enabled(False)
    model.eval()
    try:
        with torch.no_grad():
            for batch_index, (lr_img, _) in enumerate(train_loader):
                if batch_index >= args.quant_calibration_batches:
                    break
                model(lr_img.to(device).float() / 255.0, channels=args.channel_nums)
    finally:
        for handle in handles:
            handle.remove()
        model.set_quantization_enabled(True)
        model.train(was_training)

    for name, module in quantized_layers.items():
        if not samples[name]:
            raise RuntimeError(f'No calibration samples collected for {name}')
        module.act_quant.initialize_from_samples(torch.cat(samples[name], dim=1))
    logger.info(
        f'Initialized {len(quantized_layers)} LSQ+ activation quantizers with MSE calibration '
        f'from {args.quant_calibration_batches} full-precision batches.'
    )


def _optimizer_parameter_groups(model, args):
    quant_parameters = []
    quant_parameter_ids = set()
    for module in model.modules():
        if isinstance(module, QConv2dLSQP):
            for parameter in (*module.act_quant.parameters(), *module.weight_quant.parameters()):
                if id(parameter) not in quant_parameter_ids:
                    quant_parameters.append(parameter)
                    quant_parameter_ids.add(id(parameter))
    network_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in quant_parameter_ids
    ]
    return [
        {'params': network_parameters, 'lr': args.lr, 'name': 'network'},
        {'params': quant_parameters, 'lr': args.quant_lr, 'name': 'quantizer'},
    ]


def _weighted_val_loss(model, val_loaders, loss_func, device, channels):
    weighted_val_loss = 0.0
    total_val_samples = 0
    for loader in val_loaders.values():
        loader_loss = validate_epoch(
            model, loader, loss_func, device, is_residual=True, channels=channels
        )
        sample_count = len(loader.dataset)
        weighted_val_loss += loader_loss * sample_count
        total_val_samples += sample_count
    return weighted_val_loss / total_val_samples

def _validate_configurations(model, val_loaders, loss_func, args, device):
    losses = {
        label: _weighted_val_loss(model, val_loaders, loss_func, device, channels)
        for label, channels in _model_configurations(args)
    }
    full_loss = losses[f'{args.channel_nums}ch']
    return full_loss, losses


def _log_model_metrics(logger, model, val_loaders, args, device):
    for label, channels in _model_configurations(args):
        for dataset_name, loader in val_loaders.items():
            val_metrics = validate_metrics(
                model, loader, args.scale, device, 1.0, is_residual=True, channels=channels
            )
            logger.log_validation_results(f'{dataset_name}/{label}', val_metrics)


def main():

    args = train_parser()
    args.model_name = "QDPSR"
    args.is_residual = True
    # Quantization-aware training and validation use the 32-channel path only.
    args.joint_width_training = False
    args.alternate_width_training = False
    if args.subnet_loss_weight < 0.0 or args.distill_loss_weight < 0.0:
        raise ValueError('subnet and distillation loss weights must be non-negative')
    if args.quant_calibration_batches < 0 or args.quant_calibration_samples < 1:
        raise ValueError('quant calibration batches must be non-negative and samples must be positive')
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 创建logger
    logger = create_logger(log_dir="./logs", model_name=args.model_name, scale=args.scale)
    logger.info(f"使用设备: {device}")
    logger.info(f'QAT width: {args.channel_nums}ch only')

    datasets_root = Path(args.datasets_root)
    train_loader = create_train_loader(
        datasets_root / 'train',
        scale=args.scale,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        in_channels=args.in_channels
    )
                                   
    val_loaders = {
        name: create_val_loader(
            datasets_root / 'val' / name,
            args.scale,
            in_channels=args.in_channels,
        )
        for name in ('Set5', 'Set14', 'B100', 'U100', 'M109')
    }
    
    time_stamp = datetime.now().strftime("%m%d_%H%M")
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{args.model_name}_x{args.scale}_{time_stamp}.pth"
    model = _build_q_model(args, device)

    if args.pretrained_fp:
        _load_fp_checkpoint_into_q_model(model, args.pretrained_fp, device, logger)

    _calibrate_lsqplus_activations(model, train_loader, args, device, logger)

    # 统计模型参数量
    total_params = model.param_num()
    logger.info(f"模型总参数量: {total_params:,}")

    # 损失函数
    loss_func = MixedLoss(eps=1e-8, gamma=0.2)

    # 优化器
    optimizer = optim.Adam(_optimizer_parameter_groups(model, args), betas=(0.9, 0.999))
    logger.info(f'Optimizer learning rates: network={args.lr:.3e}, quantizer={args.quant_lr:.3e}')
    
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    
    # 学习率调度器：warmup + cosine annealing
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        eta_min=args.minlr,
        warmup_start_lr=3e-4
    )
   
    # 记录训练开始信息
    logger.log_training_start(args, total_params, len(train_loader))

    # 训练循环
    best_val_loss = float('inf')

    logger.info("Begin Training")
    for epoch in range(args.epochs):
        # 训练
        train_loss = train_epoch(
            model,
            train_loader,
            loss_func,
            optimizer,
            device,
            epoch,
            ema=ema,
            is_residual=True,
            joint_width_training=args.joint_width_training,
            alternate_width_training=args.alternate_width_training,
            subnet_channels=args.subnet_channels,
            subnet_loss_weight=args.subnet_loss_weight,
            distill_loss_weight=args.distill_loss_weight,
        )

        current_lr = optimizer.param_groups[0]['lr']
        
        logger.log_epoch_train(epoch, args.epochs, train_loss, current_lr)

        best_candidate = None
        with ema.average_parameters():
            val_loss, validation_losses = _validate_configurations(
                model, val_loaders, loss_func, args, device
            )

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
                'scheduler_state_dict': scheduler.state_dict(),
                'validation_losses': validation_losses,
                'model_config': {
                    'quantizer': 'lsqplus',
                    'quantizer_version': 'per_channel_projected_step_v3',
                    'quant_lr': args.quant_lr,
                    'width_training': 'full_only',
                    'scale': args.scale,
                    'in_channels': args.in_channels,
                    'num_blocks': args.num_blocks,
                    'full_channels': args.channel_nums,
                    'subnet_channels': args.subnet_channels,
                    'wbits': args.wbits,
                    'abits': args.abits,
                },
            }, model_path)
            
            logger.log_best_model(val_loss)
            _log_model_metrics(logger, best_candidate, val_loaders, args, device)

        scheduler.step()
    
    logger.log_training_finished()

    logger.log_testing_start("Best Model")
    net = _build_q_model(args, device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    net.load_state_dict(state_dict['model_state_dict'])
    net.eval()
    _log_model_metrics(logger, net, val_loaders, args, device)

    logger.log_testing_start("Bilinear Interpolation")
    for dataset_name, loader in val_loaders.items():
        val_metrics = bilinear_metrics(loader, args.scale, device)
        logger.log_validation_results(dataset_name, val_metrics)

    # 关闭logger
    logger.close()

if __name__ == "__main__":
    main()
