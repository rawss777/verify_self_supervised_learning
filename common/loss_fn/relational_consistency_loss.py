import torch
from torch import nn
from torch.nn import functional as F


class RelationalConsistencyLoss(object):
    def __init__(self, t_query: float, t_key: float, **kwargs):
        super(RelationalConsistencyLoss).__init__()
        self.t_query = t_query
        self.t_key = t_key

    def __call__(self, zq, zk, queue=None):
        if queue is None:
            return 0.

        zq = F.normalize(zq, dim=-1, p=2)  # normalize
        zk = F.normalize(zk, dim=-1, p=2)  # normalize
        queue = F.normalize(queue, dim=-1, p=2)  # normalize

        logit_q = torch.mm(zq, queue.t()) / self.t_query
        logit_k = torch.mm(zk, queue.t()).detach() / self.t_key

        loss = -torch.sum(
            F.softmax(logit_k, dim=-1) * F.log_softmax(logit_q, dim=-1),
            dim=1).mean()
        return loss


