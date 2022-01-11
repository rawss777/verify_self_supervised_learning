"""
Layer-wise adaptive rate scaling for SGD in PyTorch!
Based on https://github.com/noahgolmant/pytorch-lars
"""
import torch
from torch.nn import BatchNorm1d, BatchNorm2d
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        eta (float, optional): LARS coefficient
    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888
    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=1.0, momentum=0.9, weight_decay=0.0005, eta=0.001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=eta,
            use_lars=True,
        )
        super().__init__(params, defaults)

    def step(self, epoch=None, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            epoch: current epoch to calculate polynomial LR decay schedule.
                   if None, uses self.epoch and increments it.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eta = group["eta"]
            global_lr = group["lr"]
            use_lars = group["use_lars"]
            group["lars_lrs"] = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p)

                # Update the momentum term
                if use_lars:
                    # Compute local learning rate for this layer
                    local_lr = eta * weight_norm / (grad_norm + weight_decay * weight_norm)
                    actual_lr = local_lr * global_lr
                    group["lars_lrs"].append(actual_lr.item())
                else:
                    actual_lr = global_lr
                    group["lars_lrs"].append(global_lr)

                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                else:
                    buf = param_state["momentum_buffer"]

                buf.mul_(momentum).add_(d_p + weight_decay * p.data, alpha=actual_lr)
                p.data.add_(-buf)

        return loss


# all bias and bn weight is normal mSGD, the other is LARS
def __configure_params(model, exclude_module=[BatchNorm1d, BatchNorm2d], exclude_parma_name=['bias']):
    regular_parameters = []
    regular_parameter_names = []
    excluded_parameters = []
    excluded_parameter_names = []

    excluded_modules_name = []
    for name, modules in model.named_modules():
        if isinstance(modules, tuple(exclude_module)):
            excluded_modules_name.append(name)

    for name, params in model.named_parameters():
        post_name = name.split('.')[-1]
        if any(x in post_name for x in exclude_parma_name) or any(x in name for x in excluded_modules_name):
            excluded_parameter_names.append(name)
            excluded_parameters.append(params)
        else:
            regular_parameter_names.append(name)
            regular_parameters.append(params)

    param_groups = [
        {"params": regular_parameters, "names": regular_parameter_names, "use_lars": True},
        {"params": excluded_parameters, "names": excluded_parameter_names, "use_lars": False, "weight_decay": 0,},
    ]
    return param_groups

