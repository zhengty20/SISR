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

class WarmupCosineScheduler:
    """Linear warmup followed by CosineAnnealingLR."""
    def __init__(self, optimizer, total_epochs, warmup_epochs=0, eta_min=0.0, warmup_start_lr=1e-7):
        self.optimizer = optimizer
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.eta_min = eta_min
        self.warmup_start_lr = warmup_start_lr
        self.current_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.cosine_scheduler = None

        if self.total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.warmup_epochs >= self.total_epochs:
            raise ValueError("warmup_epochs must be smaller than total_epochs")

        if self.warmup_epochs > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.warmup_start_lr

    def _build_cosine(self):
        if self.cosine_scheduler is None:
            cosine_epochs = self.total_epochs - self.warmup_epochs
            self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_epochs,
                eta_min=self.eta_min
            )

    def step(self):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.warmup_start_lr + (self.base_lrs[i] - self.warmup_start_lr) * progress
            return

        self._build_cosine()
        self.cosine_scheduler.step()

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        state = {
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'warmup_epochs': self.warmup_epochs,
            'eta_min': self.eta_min,
            'warmup_start_lr': self.warmup_start_lr,
            'base_lrs': self.base_lrs,
        }
        if self.cosine_scheduler is not None:
            state['cosine_state'] = self.cosine_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']
        self.total_epochs = state_dict['total_epochs']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.eta_min = state_dict['eta_min']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        self.base_lrs = state_dict['base_lrs']

        if 'cosine_state' in state_dict:
            self._build_cosine()
            self.cosine_scheduler.load_state_dict(state_dict['cosine_state'])
