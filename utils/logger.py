import logging
import os
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    def __init__(self, log_dir="./logs", experiment_name=None):
        """
        初始化训练日志记录器
        
        Args:
            log_dir (str): 日志文件保存目录
            experiment_name (str): 实验名称，如果为None则使用时间戳
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果没有指定实验名称，使用时间戳
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M")
        
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.log"
        
        # 设置logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 创建文件handler
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # 创建控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 创建formatter
            formatter = logging.Formatter('%(message)s')
            
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加handler到logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def info(self, message):
        """记录INFO级别日志"""
        self.logger.info(message)
    
    def warning(self, message):
        """记录WARNING级别日志"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录ERROR级别日志"""
        self.logger.error(message)
    
    def debug(self, message):
        """记录DEBUG级别日志"""
        self.logger.debug(message)
    
    def log_training_start(self, args, total_params, num_train_batches, num_val_batches):
        """记录训练开始信息"""
        self.info("=" * 50)
        self.info("Training Started")
        self.info("=" * 50)
        self.info(f"实验名称: {self.experiment_name}")
        self.info(f"放大倍数: {args.scale}")
        self.info(f"批次大小: {args.batch_size}")
        self.info(f"学习率: {args.lr}")
        self.info(f"最小学习率: {args.minlr}")
        self.info(f"训练轮数: {args.epochs}")
        self.info(f"设备: {args.device}")
        self.info(f"模型参数数量: {total_params:,}")
        self.info(f"训练批次: {num_train_batches}")
        self.info(f"验证批次: {num_val_batches}")
        self.info("=" * 50)
    
    def log_epoch_train(self, epoch, total_epochs, train_loss, lr):
        """记录每轮训练信息"""
        self.info(f"Epoch {epoch+1}/{total_epochs}: Training loss={train_loss:.6f}, lr={lr:.2e}")
    
    def log_epoch_val(self, epoch, total_epochs, val_loss):
        """记录每轮验证信息"""
        self.info(f"Epoch {epoch+1}/{total_epochs}: Validation loss={val_loss:.6f}")
    
    def log_best_model(self, loss):
        """记录最佳模型保存信息"""
        self.info(f"Model saved! Best loss: {loss:.6f}")
    
    def log_validation_results(self, dataset_name, metrics):
        """记录验证结果"""
        self.info(f"{dataset_name}: PSNR(Y)={metrics['psnr']:.3f}, SSIM(Y)={metrics['ssim']:.4f}")
    
    def log_training_finished(self):
        """记录训练结束"""
        self.info("=" * 50)
        self.info("Training Finished")
        self.info("=" * 50)
    
    def log_testing_start(self, model_type="Best Model"):
        """记录测试开始"""
        self.info("=" * 50)
        self.info(f"Testing {model_type}")
        self.info("=" * 50)
    
    def close(self):
        """关闭logger"""
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)


def create_logger(log_dir="./logs", model_name=None, scale=None):
    """
    创建logger实例的便捷函数
    
    Args:
        log_dir (str): 日志目录
        experiment_name (str): 实验名称
        scale (int): 放大倍数，会自动添加到实验名称中
    
    Returns:
        TrainingLogger: logger实例
    """
    timestamp = datetime.now().strftime("%m%d_%H%M")
    experiment_name = f"{model_name}_x{scale}_{timestamp}"
    
    return TrainingLogger(log_dir, experiment_name) 