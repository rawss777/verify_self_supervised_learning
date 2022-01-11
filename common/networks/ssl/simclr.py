import torch
from torch import nn

from .base_ssl import BaseSSL
from .util_modules import MLP


class SimCLR(BaseSSL):
    def __init__(self, encoder: nn.Module, criterion: nn.Module, device: str,
                 proj_params: dict):
        # define encoder, criterion, device
        super().__init__(encoder, criterion, device)
        self.projector = MLP(input_dim=self.encoder.output_dim, **proj_params)

    def forward(self, multi_img_list: list, *args, **kwargs):
        x = torch.cat(multi_img_list, dim=0)
        x = x.to(self.device, non_blocking=True)

        rep = self.encoder(x)
        proj = self.projector(rep)

        loss = self.criterion(proj, negative_feature=None)

        return loss

