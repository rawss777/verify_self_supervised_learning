import torch
from torch import nn

from .base_ssl import BaseSSL
from .util_modules import MLP


class SimCLR(BaseSSL):
    def __init__(self, encoder: nn.Module, device: str, loss_params: dict,
                 proj_params: dict):
        # define query encoder, device, criterion
        super().__init__(encoder, device, loss_params)

        self.projector = MLP(input_dim=self.encoder.output_dim, **proj_params)

    def forward(self, multi_img_list: list, *args, **kwargs):
        x = torch.cat(multi_img_list, dim=0)
        x = x.to(self.device, non_blocking=True)

        proj = self.projector(self.encoder(x))
        proj_list = proj.chunk(2)

        loss = self.criterion(proj_list, negative_feature=None)

        return loss

