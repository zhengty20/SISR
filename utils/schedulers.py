from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch
import math
from collections import Counter, defaultdict

class KneeLRScheduler(_LRScheduler):
    def __init__(self, optimizer, peak_lr, warmup_steps=0, explore_steps=0, total_steps=0, min_lr=0):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.explore_steps = explore_steps
        self.total_steps = total_steps
        self.decay_steps = self.total_steps - (self.explore_steps + self.warmup_steps)
        self.current_step = 1
        self.min_lr = min_lr

        assert self.decay_steps >= 0

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(self.current_step)

    def get_lr(self, global_step):
        if global_step <= self.warmup_steps:
            return self.peak_lr * global_step / self.warmup_steps
        elif global_step <= (self.explore_steps + self.warmup_steps):
            return self.peak_lr
        else:
            slope = -1 * self.peak_lr / self.decay_steps
            return max(self.min_lr, self.peak_lr + slope * (global_step - (self.explore_steps + self.warmup_steps)))

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def step(self):
        self.current_step += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(self.current_step)

class StepLRScheduler:
    def __init__(self, optimizer, peak_lr, gamma=0.5, step_size=200, max_epochs=1000):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.step_size = step_size
        self.current_step = 0

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(self.current_step)

        if not isinstance(self.optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(self.optimizer).__name__))

    def get_lr(self, global_step):
        factor = global_step // self.step_size
        lr = self.peak_lr * (self.gamma ** factor)
        return lr

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def step(self):
        self.current_step += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(self.current_step)

class MultiStepLR_Restart(_LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            restarts=None,
            weights=None,
            gamma=0.1,
            clear_state=False,
            last_epoch=-1,
    ):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.clear_state = clear_state
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        assert len(self.restarts) == len(
            self.restart_weights
        ), "restarts and their weights do not match."
        super(MultiStepLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            # print(self.optimizer.param_groups)
            return [
                group["initial_lr"] * weight for group in self.optimizer.param_groups
            ]
        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            group["lr"] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def step(self):
        self.current_step += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(self.current_step)

class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(
            self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1
    ):
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        self.current_step = 1
        assert len(self.restarts) == len(
            self.restart_weights
        ), "restarts and their weights do not match."
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [
                group["initial_lr"] * weight for group in self.optimizer.param_groups
            ]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (
                2 * self.T_max
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max))
            / (
                    1
                    + math.cos(
                math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max
            )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

class KneeLRScheduler_Restart(_LRScheduler):
    def __init__(self, optimizer, peak_lr, warmup_steps=0, explore_steps=0, 
                 total_steps=0, min_lr=0, restarts=None, weights=None, 
                 clear_state=False, weight_decay_factor=None, last_epoch=-1):
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.explore_steps = explore_steps
        self.total_steps = total_steps
        self.decay_steps = self.total_steps - (self.explore_steps + self.warmup_steps)
        self.min_lr = min_lr
        self.restarts = restarts if restarts else []
        self.clear_state = clear_state
        self.last_restart = 0
        self.restart_cycles = 0  # 添加重启计数器
        
        # 自动生成递减权重（如每次restart权重减半）
        if weights is None and weight_decay_factor is not None and restarts:
            self.restart_weights = [weight_decay_factor ** i for i in range(len(restarts))]
        else:
            self.restart_weights = weights if weights else []
        
        assert self.decay_steps >= 0
        assert len(self.restarts) == len(self.restart_weights), "restarts and their weights do not match."
        
        super(KneeLRScheduler_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.peak_lr / self.warmup_steps if self.warmup_steps > 0 else self.peak_lr for _ in self.optimizer.param_groups]
        elif self.last_epoch in self.restarts:
            # 清除optimizer state（包括momentum、weight decay累积等）
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            
            self.last_restart = self.last_epoch
            self.restart_cycles += 1
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            
            # 重启时，我们应该从warmup阶段开始，所以返回第一步的学习率
            current_peak_lr = self.peak_lr * weight
            if self.warmup_steps > 0:
                # 重启后的第一步应该是warmup的第一步
                return [current_peak_lr / self.warmup_steps for _ in self.optimizer.param_groups]
            else:
                return [current_peak_lr for _ in self.optimizer.param_groups]
        else:
            # 计算相对于上次重启的步数
            relative_step = self.last_epoch - self.last_restart
            
            # 获取当前周期的effective peak_lr
            if self.last_restart in self.restarts:
                weight_idx = self.restarts.index(self.last_restart)
                weight = self.restart_weights[weight_idx]
                current_peak_lr = self.peak_lr * weight
            else:
                current_peak_lr = self.peak_lr
            
            if relative_step <= self.warmup_steps:
                # Warmup阶段：从接近0开始到current_peak_lr
                if self.warmup_steps > 0:
                    lr = current_peak_lr * relative_step / self.warmup_steps
                else:
                    lr = current_peak_lr
            elif relative_step <= (self.explore_steps + self.warmup_steps):
                # Explore阶段：保持current_peak_lr
                lr = current_peak_lr
            else:
                # Decay阶段：从current_peak_lr线性下降到min_lr
                decay_step = relative_step - (self.explore_steps + self.warmup_steps)
                if self.decay_steps > 0:
                    slope = -1 * (current_peak_lr - self.min_lr) / self.decay_steps
                    lr = max(self.min_lr, current_peak_lr + slope * decay_step)
                else:
                    lr = self.min_lr
            
            return [lr for _ in self.optimizer.param_groups]

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
class WarmupPlateauScheduler:
    def __init__(self, optimizer, warmup_epochs, plateau_scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.plateau_scheduler = plateau_scheduler
        self.current_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, metrics=None):
        self.current_epoch += 1
        
        # Warmup 阶段：线性增加学习率
        if self.current_epoch <= self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * 0.1 * progress
            return
        
        # Plateau 阶段：使用 ReduceLROnPlateau
        if metrics is not None:
            self.plateau_scheduler.step(metrics)
        else:
            # 如果没有提供指标，则按 epoch 更新（不推荐）
            self.plateau_scheduler.step()

class ImprovedWarmupPlateauScheduler:
    """
    改进的Warmup + Plateau学习率调度器
    特点：
    1. 支持多种warmup策略（线性、余弦、指数）
    2. 更灵活的warmup配置
    3. 完整的状态保存和恢复
    4. 更好的学习率调整策略
    """
    def __init__(self, optimizer, warmup_epochs, plateau_scheduler, 
                 warmup_start_lr=1e-7, warmup_strategy='linear', 
                 verbose=False):
        """
        Args:
            optimizer: PyTorch优化器
            warmup_epochs: Warmup阶段的epoch数量
            plateau_scheduler: ReduceLROnPlateau调度器实例
            warmup_start_lr: Warmup开始时的学习率（默认很小的值）
            warmup_strategy: Warmup策略 ('linear', 'cosine', 'exponential')
            verbose: 是否打印详细信息
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.plateau_scheduler = plateau_scheduler
        self.warmup_start_lr = warmup_start_lr
        self.warmup_strategy = warmup_strategy
        self.verbose = verbose
        
        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
        self.warmup_finished = False
        
        # 验证warmup策略
        valid_strategies = ['linear', 'cosine', 'exponential']
        if warmup_strategy not in valid_strategies:
            raise ValueError(f"warmup_strategy must be one of {valid_strategies}")
            
        # 初始化学习率为warmup起始值
        if warmup_epochs > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_start_lr
    
    def _get_warmup_lr(self, epoch):
        """根据不同策略计算warmup阶段的学习率"""
        if self.warmup_epochs == 0:
            return self.base_lrs
        
        progress = epoch / self.warmup_epochs
        lrs = []
        
        for base_lr in self.base_lrs:
            if self.warmup_strategy == 'linear':
                # 线性增长：从warmup_start_lr到base_lr
                lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress
            elif self.warmup_strategy == 'cosine':
                # 余弦增长：平滑的S形曲线
                lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (1 - math.cos(math.pi * progress)) / 2
            elif self.warmup_strategy == 'exponential':
                # 指数增长：慢启动，后期快速增长
                lr = self.warmup_start_lr * (base_lr / self.warmup_start_lr) ** progress
            else:
                lr = base_lr
            
            lrs.append(lr)
        
        return lrs
    
    def step(self, metrics=None):
        """更新学习率"""
        self.current_epoch += 1
        
        # Warmup阶段
        if self.current_epoch <= self.warmup_epochs:
            lrs = self._get_warmup_lr(self.current_epoch)
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = lrs[i]
            
            if self.verbose:
                print(f"Warmup Epoch {self.current_epoch}/{self.warmup_epochs}, "
                      f"LR: {lrs[0]:.2e} ({self.warmup_strategy} strategy)")
            return
        
        # 标记warmup结束
        if not self.warmup_finished:
            self.warmup_finished = True
            # 确保学习率设置为目标值
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i]
            if self.verbose:
                print(f"Warmup完成，切换到Plateau调度器，LR: {self.base_lrs[0]:.2e}")
        
        # Plateau阶段：使用ReduceLROnPlateau
        if metrics is not None:
            old_lr = self.optimizer.param_groups[0]['lr']
            self.plateau_scheduler.step(metrics)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if self.verbose and old_lr != new_lr:
                print(f"Plateau调度器更新学习率：{old_lr:.2e} -> {new_lr:.2e}")
        else:
            if self.verbose:
                print("警告：Plateau阶段需要提供metrics参数")
    
    def get_lr(self):
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """保存调度器状态"""
        return {
            'current_epoch': self.current_epoch,
            'warmup_finished': self.warmup_finished,
            'base_lrs': self.base_lrs,
            'warmup_epochs': self.warmup_epochs,
            'warmup_start_lr': self.warmup_start_lr,
            'warmup_strategy': self.warmup_strategy,
            'plateau_state': self.plateau_scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.current_epoch = state_dict['current_epoch']
        self.warmup_finished = state_dict['warmup_finished']
        self.base_lrs = state_dict['base_lrs']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        self.warmup_strategy = state_dict['warmup_strategy']
        self.plateau_scheduler.load_state_dict(state_dict['plateau_state'])
    
    def get_last_lr(self):
        """获取上一次的学习率（兼容PyTorch的_LRScheduler接口）"""
        return self.get_lr()


class AdaptiveWarmupPlateauScheduler:
    """
    自适应Warmup + Plateau学习率调度器
    特点：
    1. 根据训练损失自动调整warmup长度
    2. 动态监控训练稳定性
    3. 支持early stopping for warmup
    """
    def __init__(self, optimizer, max_warmup_epochs, plateau_scheduler,
                 warmup_start_lr=1e-7, loss_threshold=0.1, 
                 stable_epochs=3, verbose=False):
        """
        Args:
            optimizer: PyTorch优化器
            max_warmup_epochs: 最大warmup epoch数
            plateau_scheduler: ReduceLROnPlateau调度器实例
            warmup_start_lr: Warmup开始时的学习率
            loss_threshold: 损失稳定性阈值
            stable_epochs: 连续稳定epoch数量
            verbose: 是否打印详细信息
        """
        self.optimizer = optimizer
        self.max_warmup_epochs = max_warmup_epochs
        self.plateau_scheduler = plateau_scheduler
        self.warmup_start_lr = warmup_start_lr
        self.loss_threshold = loss_threshold
        self.stable_epochs = stable_epochs
        self.verbose = verbose
        
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
        self.warmup_finished = False
        self.loss_history = []
        self.stable_count = 0
        
        # 初始化学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_start_lr
    
    def _is_training_stable(self, current_loss):
        """检查训练是否稳定"""
        if len(self.loss_history) < 2:
            return False
        
        # 计算最近几个epoch的损失变化
        recent_losses = self.loss_history[-self.stable_epochs:]
        if len(recent_losses) < self.stable_epochs:
            return False
        
        # 检查损失是否在阈值范围内稳定
        loss_changes = [abs(recent_losses[i] - recent_losses[i-1]) / recent_losses[i-1] 
                       for i in range(1, len(recent_losses))]
        
        return all(change < self.loss_threshold for change in loss_changes)
    
    def step(self, train_loss=None, val_metrics=None):
        """更新学习率"""
        self.current_epoch += 1
        
        if train_loss is not None:
            self.loss_history.append(train_loss)
        
        # Warmup阶段
        if not self.warmup_finished and self.current_epoch <= self.max_warmup_epochs:
            # 线性增长学习率
            progress = self.current_epoch / self.max_warmup_epochs
            lrs = [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress 
                   for base_lr in self.base_lrs]
            
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = lrs[i]
            
            # 检查是否可以提前结束warmup
            if (train_loss is not None and 
                self.current_epoch >= self.stable_epochs and 
                self._is_training_stable(train_loss)):
                self.stable_count += 1
                if self.stable_count >= self.stable_epochs:
                    self.warmup_finished = True
                    if self.verbose:
                        print(f"训练稳定，在第{self.current_epoch}个epoch提前结束warmup")
            else:
                self.stable_count = 0
            
            if self.verbose and not self.warmup_finished:
                print(f"Adaptive Warmup Epoch {self.current_epoch}/{self.max_warmup_epochs}, "
                      f"LR: {lrs[0]:.2e}, Loss: {train_loss:.4f if train_loss else 'N/A'}")
            return
        
        # 结束warmup阶段
        if not self.warmup_finished:
            self.warmup_finished = True
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i]
            if self.verbose:
                print(f"Warmup结束，切换到Plateau调度器")
        
        # Plateau阶段
        if val_metrics is not None:
            self.plateau_scheduler.step(val_metrics)
    
    def state_dict(self):
        """保存调度器状态"""
        return {
            'current_epoch': self.current_epoch,
            'warmup_finished': self.warmup_finished,
            'base_lrs': self.base_lrs,
            'loss_history': self.loss_history,
            'stable_count': self.stable_count,
            'plateau_state': self.plateau_scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.current_epoch = state_dict['current_epoch']
        self.warmup_finished = state_dict['warmup_finished']
        self.base_lrs = state_dict['base_lrs']
        self.loss_history = state_dict['loss_history']
        self.stable_count = state_dict['stable_count']
        self.plateau_scheduler.load_state_dict(state_dict['plateau_state'])
