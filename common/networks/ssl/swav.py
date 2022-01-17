import torch
from torch import nn
import numpy as np

from .base_ssl import BaseSSL
from .util_modules import MLP, CosineSimilarityLayer, QueueMemory, SinkhornKnopp


class SwAV(BaseSSL):
    def __init__(self, encoder: nn.Module, device: str, loss_params: dict,
                 proj_params: dict, memory_queue_params: dict, num_prototype_list: list, sinkhorn_knopp_params: dict):
        # define query encoder, device, criterion
        super().__init__(encoder, device, loss_params)

        self.num_global_view = 2
        assert self.num_global_view == 2, f"num_global_view must be 2, but set to {num_global_view}"
        self.num_prototype = len(num_prototype_list)

        # define networks
        self.projector = MLP(input_dim=self.encoder.output_dim, **proj_params)
        self.prototype_list = nn.ModuleList([CosineSimilarityLayer(proj_params["output_dim"], num_prototype)
                                             for num_prototype in num_prototype_list])
        self.sinkhorn_knopp = SinkhornKnopp(**sinkhorn_knopp_params)

        # define memory queue
        self.use_memory_queue = memory_queue_params["use"]
        if self.use_memory_queue:
            memory_size = max(memory_queue_params["memory_size"], max(num_prototype_list))
            assert memory_size >= memory_queue_params["dequeue_size"],  f"Must be memory_size > dequeue_size"
            self.memory_bank = QueueMemory(memory_size, )
            self.dequeue_size = memory_queue_params["dequeue_size"]

    def forward(self, multi_img_list: list, *args, **kwargs):
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
        for n_proto in range(self.num_prototype):
            prototype = self.prototype_list[n_proto]  # get prototype instance

            # compute assignments
            sim_global = prototype(z_global)
            sim_global_list = sim_global.chunk(self.num_global_view)
            # addition to sim_global_list for assignments
            pre_sim_global_list = [None for _ in range(self.num_global_view)]
            if self.use_memory_queue:
                if len(self.memory_bank) >= self.dequeue_size:
                    with torch.no_grad():
                        pre_z_global = self.memory_bank.dequeue(self.dequeue_size)
                        pre_sim_global = prototype(pre_z_global)
                        pre_sim_global_list = pre_sim_global.chunk(self.num_global_view)
            assignments_list = self.get_assignments(sim_global_list, pre_sim_global_list)

            # compute global loss
            loss += self.criterion(sim_global_list[0], assignments_list[1].detach())  # swap
            loss += self.criterion(sim_global_list[1], assignments_list[0].detach())  # swap

            # compute multi crop loss
            for z_crop, multi_crop_idx in zip(z_crop_list, multi_crop_idx_list):
                sim_crop = prototype(z_crop)
                sim_crop_list = sim_crop.chunk(len(multi_crop_idx))
                for idx in range(0, len(sim_crop_list), 2):
                    loss += self.criterion(sim_crop_list[idx], assignments_list[1].detach())    # swap
                    loss += self.criterion(sim_crop_list[idx+1], assignments_list[0].detach())  # swap

        loss /= (self.num_prototype * len(multi_img_list))  # averaging

        # update memory queue
        if self.use_memory_queue:
            self.memory_bank.enqueue(z_global.detach())

        return loss

    def __get_multi_crop_idx(self, multi_img_list):
        if len(multi_img_list) <= self.num_global_view:  # has no multi crop image
            return []

        img_shape_list = np.array([list(img.shape[-2:]) for img in multi_img_list[self.num_global_view:]])
        img_shape_pattern, img_shape_pattern_idx = np.unique(img_shape_list, axis=0, return_inverse=True)
        crop_idx_list = [np.where(img_shape_pattern_idx == idx)[0] + self.num_global_view
                         for idx in range(len(img_shape_pattern))]
        return crop_idx_list

    @torch.no_grad()
    def get_assignments(self, preds: list, additions: list):
        batch_size = preds[0].shape[0]

        # compute assignment
        assignment_list = []
        for pred, add in zip(preds, additions):
            if add is not None:
                pred = torch.cat([pred, add], dim=0)
            assignment = self.sinkhorn_knopp(pred.detach())[:batch_size]
            assignment_list.append(assignment)
        return assignment_list

