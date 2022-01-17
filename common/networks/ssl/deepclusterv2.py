import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base_ssl import BaseSSL
from .util_modules import MLP, CosineSimilarityLayer, KmeansClustering


class DeepClusterV2(BaseSSL):
    def __init__(self, encoder: nn.Module, device: str, loss_params: dict,
                 proj_params: dict, num_prototype_list: list, kmeans_params: dict):
        # define query encoder, device, criterion
        super().__init__(encoder, device, loss_params)

        self.num_global_view = 2
        assert self.num_global_view == 2, f"num_global_view must be 2, but set to {num_global_view}"

        self.proj_output_dim = proj_params['output_dim']
        self.num_prototype_list = num_prototype_list
        self.kmeans_params = kmeans_params

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
        # self.update_proto_period = 1
        # self.update_prototype_step = update_prototype_step

    def forward(self, multi_img_list: list, *args, **kwargs):
        sample_indices = args[0]
        multi_img_list = np.array(multi_img_list, dtype=object)

        # # # forward
        # global view
        global_img_list = multi_img_list[:self.num_global_view].tolist()  # global view is first two
        x_global = torch.cat(global_img_list, dim=0).to(self.device, non_blocking=False)
        z_global = self.projector(self.encoder(x_global))
        # crop view
        multi_crop_idx_list = self.__get_multi_crop_idx(multi_img_list)  # get same shape image's index
        z_crop_list = []
        for same_crop_size_idx in multi_crop_idx_list:
            x_crop = multi_img_list[same_crop_size_idx].tolist()
            x_crop = torch.cat(x_crop, dim=0).to(self.device, non_blocking=True)
            z_crop = self.projector(self.encoder(x_crop))
            z_crop_list.append(z_crop)

        # # # compute loss
        loss = 0
        for n_proto in range(len(self.num_prototype_list)):
            assignments = self.assignments[n_proto, sample_indices]
            prototype = self.prototype_list[n_proto]  # get prototype instance

            # compute global similar score
            sim_global = prototype(z_global)
            sim_global_list = sim_global.chunk(self.num_global_view)

            # compute global loss
            loss += self.criterion(sim_global_list[0], assignments.detach())
            loss += self.criterion(sim_global_list[1], assignments.detach())

            # compute multi crop loss
            for z_crop, multi_crop_idx in zip(z_crop_list, multi_crop_idx_list):
                sim_crop = prototype(z_crop)
                sim_crop_list = sim_crop.chunk(len(multi_crop_idx))
                for idx in range(0, len(sim_crop_list), 2):
                    loss += self.criterion(sim_crop_list[idx], assignments.detach())    # swap
                    loss += self.criterion(sim_crop_list[idx+1], assignments.detach())  # swap

        loss /= (len(self.num_prototype_list) * len(multi_img_list))  # averaging

        # update memory bank
        z_global_list = z_global.chunk(self.num_global_view)
        start_idx, end_idx = self.batch_idx * self.batch_size, (self.batch_idx + 1) * self.batch_size
        self.index_memory_list[start_idx:end_idx] = sample_indices.long().to(self.device, non_blocking=True)
        self.projection_memory_list[0][start_idx:end_idx] = z_global_list[0].detach()
        self.projection_memory_list[1][start_idx:end_idx] = z_global_list[1].detach()
        self.batch_idx += 1

        return loss

    def on_train_start(self, train_loader, *args, **kwargs):
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
        # if self.update_prototype_step[self.update_proto_period - 1] <= n_epoch:
        #     self.update_proto_period += 1

        if n_epoch == 1:
            self.assignments = -torch.ones(
                len(self.num_prototype_list), self.num_dataset, device=self.device).long()
        else:
            # if n_epoch % self.update_proto_period == 0:
            with torch.no_grad():
                centroids = []
                for prototype in self.prototype_list:
                    centroids.append(prototype.prototypes.weight.data.clone())

                centroids, self.assignments = self.clustering.fit(
                    self.projection_memory_list, self.index_memory_list, centroids)
                for prototype, centroid in zip(self.prototype_list, centroids):
                    dtype = prototype.prototypes.weight.dtype
                    prototype.prototypes.weight.copy_(centroid.to(dtype))

    def __get_multi_crop_idx(self, multi_img_list):
        if len(multi_img_list) <= self.num_global_view:  # has no multi crop image
            return []

        img_shape_list = np.array([list(img.shape[-2:]) for img in multi_img_list[self.num_global_view:]])
        img_shape_pattern, img_shape_pattern_idx = np.unique(img_shape_list, axis=0, return_inverse=True)
        crop_idx_list = [np.where(img_shape_pattern_idx == idx)[0] + self.num_global_view
                         for idx in range(len(img_shape_pattern))]
        return crop_idx_list

