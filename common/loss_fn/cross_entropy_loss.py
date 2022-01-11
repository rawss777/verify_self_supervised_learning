import torch
from torch import nn
from torch.nn import functional as F


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, n_views, device, t_scale):
        super(SoftCrossEntropyLoss, self).__init__()
        self.t_scale = t_scale

    def forward(self, logits, soft_labels):
        loss = -((soft_labels * torch.log_softmax(logits, dim=-1)) / self.t_scale).sum(dim=-1)
        return loss.mean()


class WithTempCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, n_views, device, t_scale, **kwargs):
        super(WithTempCrossEntropyLoss, self).__init__(**kwargs)
        self.t_scale = t_scale

    def forward(self, logits, labels):
        return super().forward(logits / self.t_scale, labels)


