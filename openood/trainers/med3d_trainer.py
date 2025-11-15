from functools import partial

import torch
import torch.nn.functional as F
import torch.optim as optim
from monai.losses import FocalLoss

from .base_trainer import BaseTrainer


class Med3DTrainer(BaseTrainer):
    def get_optimizer(self):
        optimizer_config = self.config.optimizer
        if optimizer_config.name.lower() == 'sgd':
            return optim.SGD(
                self.net.parameters(),
                lr=optimizer_config.lr,
                momentum=optimizer_config.momentum,
                weight_decay=optimizer_config.weight_decay,
                nesterov=True,
            )
        elif optimizer_config.name.lower() == 'adam':
            return optim.Adam(self.net.parameters(),
                              lr=optimizer_config.lr,
                              betas=optimizer_config.betas,
                              weight_decay=optimizer_config.weight_decay)
        else:
            raise ValueError('Optimizer not supported')

    def get_scheduler(self):
        scheduler_config = self.config.scheduler
        total_steps = self.config.optimizer.num_epochs * len(self.train_loader)
        if scheduler_config is None:
            return type('DummyScheduler', (), {'step': lambda *a: None})()
        elif scheduler_config.name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                        T_max=total_steps,
                                                        eta_min=1e-6)
        elif scheduler_config.name.lower() == 'warmup+cosine':
            warmup_steps = scheduler_config.warmup_epochs * len(
                self.train_loader)
            warmup_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda s: (s + 1) / warmup_steps)
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
            return optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps])
        elif scheduler_config.name.lower() == 'poly':
            return optim.lr_scheduler.PolynomialLR(
                self.optimizer,
                total_iters=total_steps,
                power=scheduler_config.power,
            )
        elif scheduler_config.name.lower() == 'multi-step':
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=scheduler_config.milestones,
                gamma=scheduler_config.gamma)
        else:
            raise ValueError('Scheduler not supported')

    def get_loss_fn(self):
        loss_config = self.config.loss
        if loss_config is None:
            return super().get_loss_fn()
        elif loss_config.name.lower() == 'cross_entropy':
            return F.cross_entropy
        elif loss_config.name.lower() == 'weighted_ce':
            weight = torch.Tensor(loss_config.weight).cuda()
            return partial(F.cross_entropy, weight=weight)
        elif loss_config.name.lower() == 'focal':
            gamma = loss_config.gamma or 2.0
            weight = torch.Tensor(loss_config.weight).cuda()
            return FocalLoss(gamma=gamma,
                             weight=weight,
                             use_softmax=True,
                             to_onehot_y=True)
        else:
            raise ValueError('Loss function not supported')
