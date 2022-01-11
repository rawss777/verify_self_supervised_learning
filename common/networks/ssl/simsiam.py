import torch
from torch import nn

from .base_ssl import BaseSSL
from .util_modules import MLP, EMA, EMAN


class SimSiam(BaseSSL):
    def __init__(self, encoder: nn.Module, criterion: nn.Module, device: str,
                 proj_params: dict, pred_params: dict):
        # define online encoder, criterion, device
        super().__init__(encoder, criterion, device)

        # define networks
        self.projector = MLP(input_dim=encoder.output_dim, **proj_params)
        self.predictor = MLP(input_dim=proj_params['output_dim'], **pred_params)

    def forward(self, multi_img_list: list, *args, **kwargs):
        x_theta, x_xi = multi_img_list[0], multi_img_list[1]
        x_theta = x_theta.to(self.device, non_blocking=True)
        x_xi = x_xi.to(self.device, non_blocking=True)

        # get projection
        z_theta = self.projector(self.encoder(x_theta))
        z_xi = self.projector(self.encoder(x_xi))

        # get prediction
        pred_theta = self.predictor(z_theta)
        pred_xi = self.predictor(z_xi)

        # compute loss
        loss_theta = self.criterion(pred_theta, z_xi.detach())
        loss_xi = self.criterion(pred_xi, z_theta.detach())
        return (loss_theta + loss_xi) / 2

