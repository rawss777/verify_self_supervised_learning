import numpy as np
import random
from copy import deepcopy
from collections import deque, OrderedDict
from scipy.sparse import csr_matrix
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_layer: int, output_dim: int, hidden_dim: int = None,
                 output_bn: bool = False, hidden_bn: bool = True):
        super(MLP, self).__init__()
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        projector_list = []
        for n in range(1, num_layer + 1):
            if n == num_layer:  # last layer
                projector_list.append(
                    (f"fc{n}", nn.Linear(input_dim, output_dim, bias=True))
                )
            else:
                if hidden_bn:
                    projector_list.extend([
                        (f"fc{n}", nn.Linear(input_dim, hidden_dim, bias=False)),
                        (f"bn{n}", nn.BatchNorm1d(hidden_dim)),
                        (f"act{n}", nn.ReLU(inplace=True))
                    ])
                else:
                    projector_list.extend([
                        (f"fc{n}", nn.Linear(input_dim, hidden_dim, bias=False)),
                        (f"act{n}", nn.ReLU(inplace=True))
                    ])
                input_dim = hidden_dim
        if output_bn: projector_list.append((f"last_bn", nn.BatchNorm1d(output_dim)))

        self.projector = nn.Sequential(OrderedDict(projector_list))

    def forward(self, rep):
        return self.projector(rep)


class EMA(nn.Module):
    def __init__(self, online_network: nn.Module, m_factor: float):
        super(EMA, self).__init__()
        assert (0. <= m_factor) and (m_factor <= 1.), 'm_factor must be (0.0, 1.0) '

        self.m_factor = m_factor
        self.online_network = online_network
        self.target_network = deepcopy(online_network)
        for param in self.target_network.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        return self.target_network(x)

    @torch.no_grad()
    def update(self):
        for param_t, param_o in zip(self.target_network.parameters(), self.online_network.parameters()):
            param_t.data = param_t.data * self.m_factor + param_o.data * (1. - self.m_factor)


class EMAN(EMA):
    def __init__(self, online_network: nn.Module, m_factor: float):
        super().__init__(online_network, m_factor)

    def forward(self, x):
        self.target_network.eval()
        return self.target_network(x)

    def update(self):
        # update parameter
        super().update()
        # update buffers
        for (key_t, param_t), (key_o, param_o) in zip(self.target_network.named_buffers(), self.online_network.named_buffers()):
            if 'num_batches_tracked' in key_o:
                param_t.data = param_o.data
            else:
                param_t.data = param_t.data * self.m_factor + param_o.data * (1. - self.m_factor)


class QueueMemory(object):
    def __init__(self, memory_size: int):
        super(QueueMemory, self).__init__()
        self.queue_memory = deque(maxlen=memory_size)

    def __len__(self):
        return len(self.queue_memory)

    def enqueue(self, x: torch.Tensor):
        self.queue_memory.extend(x.chunk(x.shape[0]))

    def dequeue(self, dequeue_size: int):
        sample = random.sample(self.queue_memory, dequeue_size)
        return torch.cat(sample, dim=0)


class CosineSimilarityLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(CosineSimilarityLayer, self).__init__()
        self.prototypes = nn.utils.weight_norm(
            nn.Linear(input_dim, output_dim, bias=False), dim=1
        )

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.prototypes(x)


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters: int = 3, epsilon: float = 0.05, use_double_prec: bool = False,
                 hard_assignment: bool = False):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.use_double_prec = use_double_prec
        self.hard_assignment = hard_assignment
        self.eps_num_stab = 1e-12

    @torch.no_grad()
    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        if self.use_double_prec:
            Q = torch.exp(Q.double() / self.epsilon).t().double()
        else:
            Q = torch.exp(Q / self.epsilon).t()

        # remove potential infs in Q
        # replace the inf entries with the max of the finite entries in Q
        mask = torch.isinf(Q)
        idx = torch.nonzero(mask)
        if len(idx) > 0:
            for i in idx:
                Q[i[0], i[1]] = 0
            m = torch.max(Q)
            for i in idx:
                Q[i[0], i[1]] = m
        sum_Q = torch.sum(Q, dtype=Q.dtype)
        Q /= sum_Q
        K, B = Q.shape
        device = Q.device

        # we follow the u, r, c and Q notations from
        # https://arxiv.org/abs/1911.05371
        r = torch.ones(K).to(device) / K
        c = torch.ones(B).to(device) / B
        if self.use_double_prec:
            r, c = r.double(), c.double()

        for _ in range(self.num_iters):
            u = torch.sum(Q, dim=1, dtype=Q.dtype)

            # for numerical stability, add a small epsilon value
            # for non-zero Q values.
            if len(torch.nonzero(u == 0)) > 0:
                Q += self.eps_num_stab
                u = torch.sum(Q, dim=1, dtype=Q.dtype)
            u = r / u

            # remove potential infs in "u"
            # replace the inf entries with the max of the finite entries in "u"
            mask = torch.isinf(u)
            ind = torch.nonzero(mask)
            if len(ind) > 0:
                for i in ind:
                    u[i[0]] = 0
                m = torch.max(u)
                for i in ind:
                    u[i[0]] = m

            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0, dtype=Q.dtype)).unsqueeze(0)
        Q = (Q / torch.sum(Q, dim=0, keepdim=True, dtype=Q.dtype)).t().float()

        if self.hard_assignment:
            index_max = torch.max(Q, dim=1)[1]
            Q.zero_()
            Q.scatter_(1, index_max.unsqueeze(1), 1)
        return Q


class SplitBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits=1, **kwargs):
        super().__init__(num_features, **kwargs)
        self.num_splits = num_splits

    def forward(self, x):
        if self.num_splits == 1:
            return super().forward(x)  # self.num_splits = 1, Normal batch normalization
        else:
            N, C, H, W = x.shape
            if self.training or not self.track_running_stats:
                running_mean_split = self.running_mean.repeat(self.num_splits)
                running_var_split = self.running_var.repeat(self.num_splits)
                outcome = F.batch_norm(
                    x.view(-1, C * self.num_splits, H, W),
                    running_mean_split,
                    running_var_split,
                    self.weight.repeat(self.num_splits),
                    self.bias.repeat(self.num_splits),
                    True,
                    self.momentum,
                    self.eps
                ).view(N, C, H, W)
                self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
                self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
                return outcome
            else:
                return F.batch_norm(
                    x, self.running_mean, self.running_var,
                    self.weight, self.bias, False, self.momentum, self.eps)


class SplitBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, num_splits=1, **kwargs):
        super().__init__(num_features, **kwargs)
        self.num_splits = num_splits

    def forward(self, x):
        if self.num_splits == 1:
            return super().forward(x)  # self.num_splits = 1, Normal batch normalization
        else:
            N, C = x.shape
            if self.training or not self.track_running_stats:
                running_mean_split = self.running_mean.repeat(self.num_splits)
                running_var_split = self.running_var.repeat(self.num_splits)
                outcome = F.batch_norm(
                    x.view(-1, C * self.num_splits),
                    running_mean_split,
                    running_var_split,
                    self.weight.repeat(self.num_splits),
                    self.bias.repeat(self.num_splits),
                    True,
                    self.momentum,
                    self.eps
                ).view(N, C)
                self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
                self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
                return outcome
            else:
                return F.batch_norm(
                    x, self.running_mean, self.running_var,
                    self.weight, self.bias, False, self.momentum, self.eps)


class KmeansClustering(object):
    def __init__(self, proj_dim: int, num_prototype_list: list, num_global_view: int, device: str, dataset_size: int,
                 num_iter: int = 10):
        self.proj_dim = proj_dim
        self.num_prototype_list = num_prototype_list
        self.num_global_view = num_global_view
        self.device = device
        self.dataset_size = dataset_size
        self.num_iter = num_iter

    @torch.no_grad()
    def fit(self, projection_list: torch.Tensor, index_list: torch.Tensor):
        """
        :param projection_list: List of one global_view's projection torch.tensor.
                                Each projection shape is (num_sample, proj_dim)
        :param index_list     : Original training sample indices seen from projection vector indices
        :return:
            centroids_list:   List of prototype(cluster) centroids.
                              Length is self.num_prototype_list.
            assignments_list: List of prototype(cluster) assignments from projection.
                              Length is self.num_prototype_list.
        """
        centroids_list = []
        n_view = 0
        assignments_list = -torch.ones(len(self.num_prototype_list), self.dataset_size).long().to(self.device)
        for target_proto_idx, num_proto in enumerate(self.num_prototype_list):
            target_centroids, target_assignments = self.__compute_centroids_and_assignments(projection_list[n_view],
                                                                                            num_proto)
            centroids_list.append(target_centroids)
            assignments_list[target_proto_idx][index_list] = target_assignments
            n_view = (n_view + 1) % self.num_global_view  # update n_view
        return centroids_list, assignments_list

    @torch.no_grad()
    def __compute_centroids_and_assignments(self, projection: torch.Tensor, num_proto: int):
        projection = F.normalize(projection, dim=1, p=2)  # normalize projection

        # centroids = torch.empty(num_proto, self.proj_dim).to(self.device, non_blocking=True)
        random_idx = torch.randperm(len(projection))[:num_proto]
        assert len(random_idx) >= num_proto, "please reduce the number of centroids"
        centroids = projection[random_idx]

        for n_iter in range(self.num_iter + 1):
            # # # E step (update assignments)
            # Compute assignments from each projection
            dot_products = torch.mm(projection, centroids.t())
            assignments = dot_products.max(dim=1)[1]

            # finish
            if n_iter == self.num_iter:
                break

            # # # M step (update centroids)
            # Compute each proto(cluster) centroid
            assignments_indices = self.get_indices_sparse(assignments.cpu().numpy())

            counts = torch.zeros(num_proto).to(self.device, non_blocking=True).long()
            proj_sums = torch.zeros(num_proto, self.proj_dim).to(self.device, non_blocking=True)
            for n_proto in range(len(assignments_indices)):
                n_proto_indices_list = assignments_indices[n_proto][0]
                if len(n_proto_indices_list) > 0:
                    proj_sums[n_proto] = torch.sum(
                        projection[n_proto_indices_list], dim=0,
                    )
                    counts[n_proto] = len(n_proto_indices_list)

            mask = counts > 0
            centroids[mask] = proj_sums[mask] / counts[mask].unsqueeze(1)
            centroids = F.normalize(centroids, dim=1, p=2)  # normalize centroids

        return centroids, assignments.long()

    @staticmethod
    def get_indices_sparse(data: np.ndarray) -> list:
        """
        各クラスタの番号(data=c)のindexを取得(np.argwhereより高速)
        retrun例(dataが2次元配列の場合):
        0_cluster_indices = data_indices[0]  # 0クラスのindiceタプル
        0_cluster_indices => tuple(axis0_indices_1darray, axis1_indices_1darray)
        """
        cols = np.arange(data.size)
        M = csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))
        data_indices = [np.unravel_index(row.data, data.shape) for row in M]
        return data_indices
