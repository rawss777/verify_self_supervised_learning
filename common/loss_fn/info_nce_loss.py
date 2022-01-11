import torch
from torch import nn
from torch.nn import functional as F


# InfoNCELoss = NTXentLoss
class InfoNCELoss(object):
    def __init__(self, n_views: int, device: str, t_scale: float):
        super(InfoNCELoss).__init__()
        self.t_scale = t_scale
        self.n_views = n_views
        self.device = device

    def __call__(self, features, negative_feature=None):
        """
        :param feature (torch.Tensor): Projected feature. Shape is ([batch_size * n_views, feature_size])
        :param negative_feature (torch.Tensor): Negative feature from queue memory (MoCov1, MoCov2, simCLRv2).
                                                Shape is ([*, feature_size])
        :return: loss values
        """
        total_batch_size = features.shape[0]

        # label (Positive pair is 1.0, negative pair is 0.0. Shape is ([total_batch_size, total_batch_size]))
        batch_size = total_batch_size // self.n_views
        label = torch.cat([torch.arange(batch_size)] * self.n_views, dim=0).to(self.device)
        label = (label.unsqueeze(0) == label.unsqueeze(1)).float()

        # similarity (Cosine similarity from feature. Shape is ([total_batch_size, total_batch_size]))
        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T)

        # Mask to same image (same image is True)
        mask = torch.eye(total_batch_size, dtype=torch.bool).to(self.device)
        label = label[~mask].view(total_batch_size, -1)
        similarity = similarity[~mask].view(total_batch_size, -1)

        # Positive
        pos = similarity[label.bool()].view(total_batch_size, -1)
        pos = torch.exp(pos / self.t_scale)

        # Negative
        neg = similarity[~label.bool()].view(total_batch_size, -1)
        if negative_feature is not None:
            negative_feature = F.normalize(negative_feature, dim=1)
            neg_similarity = torch.matmul(features, negative_feature.T)
            neg = torch.concat([neg, neg_similarity], dim=1)
        neg = torch.sum(torch.exp(neg / self.t_scale), dim=1)

        # compute cross entropy loss
        loss = torch.mean(-torch.log(pos / (pos + neg)))

        return loss


