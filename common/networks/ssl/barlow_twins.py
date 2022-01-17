import torch
from torch import nn

from .base_ssl import BaseSSL
from .util_modules import MLP, EMA, EMAN


class BarlowTwins(BaseSSL):
    def __init__(self, encoder: nn.Module, device: str, loss_params: dict,
                 proj_params: dict):
        # define online encoder, criterion, device
        super().__init__(encoder, device, loss_params)

        self.num_global_view = 2
        assert self.num_global_view == 2, f"num_global_view must be 2, but set to {num_global_view}"

        # define online networks
        self.projector = MLP(input_dim=encoder.output_dim, **proj_params)

    def forward(self, multi_img_list: list, *args, **kwargs):
        x = torch.cat(multi_img_list[:self.num_global_view], dim=0).to(self.device, non_blocking=False)
        z = self.projector(self.encoder(x))
        z1, z2 = z.chunk(2)
        loss = self.criterion(z1, z2)

        return loss

