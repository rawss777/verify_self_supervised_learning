import torch
from torch import nn
from torch.nn import functional as F


class CosineSimilarityLoss(object):
    def __init__(self, **kwargs):
        super(CosineSimilarityLoss).__init__()

    def __call__(self, feat1, feat2):
        feat1 = F.normalize(feat1, dim=-1, p=2)
        feat2 = F.normalize(feat2, dim=-1, p=2)
        return (2 - 2 * (feat1 * feat2).sum(dim=-1)).mean()


