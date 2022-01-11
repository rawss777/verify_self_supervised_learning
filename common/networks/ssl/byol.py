import torch
from torch import nn

from .base_ssl import BaseSSL
from .util_modules import MLP, EMA, EMAN


class BYOL(BaseSSL):
    def __init__(self, encoder: nn.Module, criterion: nn.Module, device: str,
                 proj_params: dict, pred_params: dict, m_factor: float, split_bn: dict, use_eman: bool):
        # define online encoder, criterion, device
        super().__init__(encoder, criterion, device)

        # define online networks
        self.projector = MLP(input_dim=encoder.output_dim, **proj_params)
        self.predictor = MLP(input_dim=proj_params['output_dim'], **pred_params)

        # define target networks
        ema_mod = EMAN if use_eman else EMA
        self.m_encoder = ema_mod(self.encoder, m_factor)
        self.m_projector = ema_mod(self.projector, m_factor)

        # replace target networks to split_bn
        self.use_split_bn = split_bn.use
        if self.use_split_bn:
            self.replace_bn_to_split_bn(self.m_encoder, split_bn.num_split)
            self.replace_bn_to_split_bn(self.m_projector, split_bn.num_split)

    def forward(self, multi_img_list: list, *args, **kwargs):
        x_theta, x_xi = multi_img_list[0], multi_img_list[1]
        x_theta = x_theta.to(self.device, non_blocking=True)
        x_xi = x_xi.to(self.device, non_blocking=True)

        # get online prediction
        online_pred_theta = self.predictor(self.projector(self.encoder(x_theta)))
        online_pred_xi = self.predictor(self.projector(self.encoder(x_xi)))

        with torch.no_grad():
            # update momentum
            self.m_encoder.update()
            self.m_projector.update()

            # batch shuffle
            if self.use_split_bn:
                x_theta, unshuffled_idx_theta = self.batch_shuffle(x_theta)
                x_xi, unshuffled_idx_xi = self.batch_shuffle(x_xi)

            # get target projection
            target_proj_theta = self.m_projector(self.m_encoder(x_theta))
            target_proj_xi = self.m_projector(self.m_encoder(x_xi))

            # batch revert
            if self.use_split_bn:
                target_proj_theta = self.batch_revert(target_proj_theta, unshuffled_idx_theta)
                target_proj_xi = self.batch_revert(target_proj_xi, unshuffled_idx_xi)

        # compute loss
        loss_theta = self.criterion(online_pred_theta, target_proj_xi.detach())
        loss_xi = self.criterion(online_pred_xi, target_proj_theta.detach())
        return (loss_theta + loss_xi) / 2

