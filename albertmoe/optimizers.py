"""
Custom optimizers for AlbertMoE models.
"""

import torch
from torch.optim.optimizer import Optimizer


class ChillAdam(Optimizer):
    def __init__(self, params, min_lr=1e-4, max_lr=1.0, eps=1e-3, betas=(0.9, 0.999), weight_decay=0, lr=1):
        """
        AdamChill: A hybrid optimizer combining Adam's momentum with ChillSGD's
        adaptive learning rate and gradient normalization.
        """
        if not 0.0 <= min_lr:
            raise ValueError(f"Invalid min_lr: {min_lr}")
        if not 0.0 <= max_lr:
            raise ValueError(f"Invalid max_lr: {max_lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(min_lr=min_lr, max_lr=max_lr, eps=eps, betas=betas, weight_decay=weight_decay, lr=lr)
        super(ChillAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            min_lr, max_lr, eps, betas, weight_decay = group['min_lr'], group['max_lr'], group['eps'], group['betas'], group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamChill does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = betas
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                grad_norm = grad.norm(p=2).clamp(min=eps, max=100.)
                grad_normalized = grad / grad_norm

                if weight_decay != 0:
                    grad_normalized = grad_normalized.add(p, alpha=weight_decay)

                exp_avg.mul_(beta1).add_(grad_normalized, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_normalized, grad_normalized, value=1 - beta2)

                param_norm = p.norm(p=2).clamp(min=eps)
                lr = 1.0 / param_norm
                lr = lr.clamp(min=min_lr, max=max_lr)
                self.state[p]["lr"] = lr.item()

                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss