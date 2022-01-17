import torch
from torch import nn
from torch.nn import functional as F


class CrossCorrelationLoss(object):
    def __init__(self, lamb: float, scale_loss: float, **kwargs):
        super(CrossCorrelationLoss).__init__()
        self.lamb = lamb
        self.scale_loss = scale_loss

    def __call__(self, feat1, feat2):
        num_feat = feat1.shape[1]
        feat1 = F.normalize(feat1, dim=0, p=2)  # normalize batch direction
        feat2 = F.normalize(feat2, dim=0, p=2)  # normalize batch direction

        corr = torch.mm(feat1.t(), feat2)
        diag = torch.eye(num_feat, device=corr.device)
        cdif = (corr - diag).pow(2)
        cdif[~diag.bool()] *= self.lamb
        loss = self.scale_loss * cdif.sum()
        return loss


