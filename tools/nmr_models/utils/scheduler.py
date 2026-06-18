"""
Learning rate schedulers for NMR foundation model training.
"""
import torch
from torch.optim.lr_scheduler import _LRScheduler


class NoamScheduler(_LRScheduler):
    """
    Noam learning rate scheduler from "Attention is All You Need" paper.

    The learning rate increases linearly during warmup, then decreases proportionally
    to the inverse square root of the step number.

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Args:
        optimizer: Wrapped optimizer
        n_warmup_steps: Number of warmup steps
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(self, optimizer, n_warmup_steps, min_lr=1e-5, last_epoch=-1):
        self.n_warmup_steps = n_warmup_steps
        self.min_lr = min_lr
        super(NoamScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate based on current step."""
        step_num = max(1, self.last_epoch + 1)

        # Noam decay formula
        scale = self.n_warmup_steps ** 0.5 * min(
            step_num ** (-0.5),
            step_num * self.n_warmup_steps ** (-1.5)
        )
        lrs = [base_lr * scale for base_lr in self.base_lrs]
        #lrs = [max(base_lr * scale, self.min_lr) for base_lr in self.base_lrs]
        return lrs


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.

    Args:
        optimizer: Wrapped optimizer
        n_warmup_steps: Number of warmup steps
        n_total_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of base_lr (default: 0.0)
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(self, optimizer, n_warmup_steps, n_total_steps, min_lr_ratio=0.0, last_epoch=-1):
        self.n_warmup_steps = n_warmup_steps
        self.n_total_steps = n_total_steps
        self.min_lr_ratio = min_lr_ratio
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate based on current step."""
        step_num = max(1, self.last_epoch + 1)

        if step_num < self.n_warmup_steps:
            # Linear warmup
            scale = step_num / self.n_warmup_steps
        else:
            # Cosine annealing
            progress = (step_num - self.n_warmup_steps) / max(1, self.n_total_steps - self.n_warmup_steps)
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
            scale = float(scale)

        return [base_lr * scale for base_lr in self.base_lrs]
