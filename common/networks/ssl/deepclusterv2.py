import torch
from torch import nn
from torch.nn import functional as F

from .base_ssl import BaseSSL
from .util_modules import MLP, CosineSimilarityLayer, KmeansClustering


class DeepClusterV2(BaseSSL):
    def __init__(self, encoder: nn.Module, criterion: nn.Module, device: str,
                 proj_params: dict, prototype_schedule: list, num_prototype_list: list, kmeans_params: dict):
        # define online encoder, criterion, device
        super().__init__(encoder, criterion, device)
        self.proj_output_dim = proj_params['output_dim']
        self.num_global_view = 2
        assert self.num_global_view == 2, f"num_global_view must be 2, but set to {num_global_view}"
        self.num_prototype_list = num_prototype_list
        self.kmeans_params = kmeans_params
        self.prototype_schedule = prototype_schedule
        self.update_step = 1

        # define online networks
        self.projector = MLP(input_dim=encoder.output_dim, **proj_params)
        self.prototype_list = nn.ModuleList([CosineSimilarityLayer(self.proj_output_dim, num_prototype)
                                             for num_prototype in self.num_prototype_list])
        # set requires grad to false
        for proto in self.prototype_list:
            for params in proto.parameters():
                params.requires_grad = False

        # define in on_train_start
        self.batch_size = None
        self.num_train_sample = None
        self.num_dataset = None
        self.clustering = None
        # define in on_train_epoch_start
        self.assignments = None
        self.batch_idx = 0

    def forward(self, multi_img_list: list, *args, **kwargs):
        sample_indices = args[0]
        x1, x2 = multi_img_list[0], multi_img_list[1]
        x1 = x1.to(self.device, non_blocking=True)
        x2 = x2.to(self.device, non_blocking=True)

        # get projection
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        proto1_list = [prototype(z1) for prototype in self.prototype_list]
        proto2_list = [prototype(z2) for prototype in self.prototype_list]
        assignments_list = self.assignments[:, sample_indices]

        # compute loss
        loss = 0.
        for proto1, proto2, assignments in zip(proto1_list, proto2_list, assignments_list):
            loss += self.criterion(proto1, assignments)
            loss += self.criterion(proto2, assignments)
        loss /= len(assignments_list)

        # update memory bank
        start_idx, end_idx = self.batch_idx * self.batch_size, (self.batch_idx + 1) * self.batch_size
        self.index_memory_list[start_idx:end_idx] = sample_indices.long().to(self.device, non_blocking=True)
        self.projection_memory_list[0][start_idx:end_idx] = z1.detach()
        self.projection_memory_list[1][start_idx:end_idx] = z2.detach()
        self.batch_idx += 1

        return loss

    def on_train_start(self, train_loader):
        self.batch_size = train_loader.batch_size
        self.num_train_sample = len(train_loader) * self.batch_size
        self.num_dataset = len(train_loader.dataset)

        # define clustering
        self.clustering = KmeansClustering(self.proj_output_dim, self.num_prototype_list, self.num_global_view,
                                           self.device, self.num_dataset, **self.kmeans_params)

        # define memory
        self.register_buffer(
            "projection_memory_list",
            F.normalize(torch.randn(self.num_global_view, self.num_train_sample, self.proj_output_dim), dim=-1
                        ).to(self.device, non_blocking=True)
        )

        self.register_buffer(
            "index_memory_list",
            torch.zeros(self.num_train_sample).long().to(self.device, non_blocking=True),
        )

    def on_train_epoch_start(self, n_epoch):
        self.batch_idx = 0
        if (n_epoch-1) > self.prototype_schedule[self.update_step-1]:
            self.update_step += 1

        if n_epoch == 1:
            self.assignments = -torch.ones(
                len(self.num_prototype_list), self.num_dataset, device=self.device).long()
        else:
            if n_epoch % self.update_step == 0:
                with torch.no_grad():
                    centroids, self.assignments = self.clustering.fit(
                        self.projection_memory_list, self.index_memory_list)
                    for prototype, centroid in zip(self.prototype_list, centroids):
                        dtype = prototype.prototypes.weight.dtype
                        prototype.prototypes.weight.copy_(centroid.to(dtype))

