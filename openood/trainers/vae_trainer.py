import numpy as np
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

import openood.utils.comm as comm


class VAETrainer:
    def __init__(self, net, feat_loader, config) -> None:
        self.config = config
        self.net = net
        self.feat_agg = net.get('feat_agg', nn.Identity())
        self.vae = net['vae']
        self.feat_loader = feat_loader

        # Freeze backbone
        for p in self.net['backbone'].parameters():
            p.requires_grad = False

        self.learnable_params = (list(self.vae.parameters()) +
                                 list(self.feat_agg.parameters()))

        optimizer_config = self.config.optimizer
        self.grad_clip_norm = optimizer_config.grad_clip_norm
        # self.beta_vae = optimizer_config.beta_vae

        self.optimizer = optim.Adam(self.learnable_params,
                                    lr=optimizer_config.lr,
                                    betas=optimizer_config.betas,
                                    weight_decay=optimizer_config.weight_decay)

    def train_epoch(self, epoch_idx):
        self.vae.train()
        self.feat_agg.train()

        feat_dataiter = iter(self.feat_loader)

        loss_avg = 0.0
        recon_loss_avg = 0.0
        kl_loss_avg = 0.0

        for train_step in tqdm(range(1,
                                     len(feat_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            feats = next(feat_dataiter)['data'].cuda()

            self.vae.zero_grad()
            self.feat_agg.zero_grad()

            # Aggregate features
            feats = self.feat_agg(feats.flatten(1))

            # Compute ELBO loss for training (returns batch means)
            loss, recon_loss_mean, kl_loss_mean = self.vae.forward_elbo(feats)

            # Backward pass
            loss.backward()

            if self.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.learnable_params,
                                               max_norm=self.grad_clip_norm,
                                               error_if_nonfinite=True)

            self.optimizer.step()

            # Exponential moving average for smooth metrics
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2
                recon_loss_avg = recon_loss_avg * 0.8 + float(
                    recon_loss_mean) * 0.2
                kl_loss_avg = kl_loss_avg * 0.8 + float(kl_loss_mean) * 0.2

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)
        metrics['recon_loss'] = self.save_metrics(recon_loss_avg)
        metrics['kl_loss'] = self.save_metrics(kl_loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])
        return total_losses_reduced
