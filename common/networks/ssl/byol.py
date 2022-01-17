import torch
from torch import nn

from .base_ssl import BaseSSL
from .util_modules import MLP, EMA, EMAN


class BYOL(BaseSSL):
    def __init__(self, encoder: nn.Module, device: str, loss_params: dict,
                 proj_params: dict, pred_params: dict, m_factor: dict, split_bn: dict, use_eman: bool):
        # define online encoder, criterion, device
        super().__init__(encoder, device, loss_params)

        # define online networks
        self.projector = MLP(input_dim=encoder.output_dim, **proj_params)
        self.predictor = MLP(input_dim=self.projector.output_dim, **pred_params)

        # define target networks
        ema_mod = EMAN if use_eman else EMA
        self.m_encoder = ema_mod(self.encoder, **m_factor)
        self.m_projector = ema_mod(self.projector, **m_factor)

        # replace target networks to split_bn
        self.use_split_bn = split_bn.use
        if self.use_split_bn:
            self.replace_bn_to_split_bn(self.m_encoder, split_bn.num_split)
            self.replace_bn_to_split_bn(self.m_projector, split_bn.num_split)

        # define in on_train_start
        self.max_iter = None
        self.iter_num_in_epoch = None
        # define in on_train_epoch_start
        self.cur_iter = 0

    def forward(self, multi_img_list: list, *args, **kwargs):
        x_theta, x_xi = multi_img_list[0], multi_img_list[1]
        x_theta = x_theta.to(self.device, non_blocking=True)
        x_xi = x_xi.to(self.device, non_blocking=True)

        # get online prediction
        online_pred_theta = self.predictor(self.projector(self.encoder(x_theta)))
        online_pred_xi = self.predictor(self.projector(self.encoder(x_xi)))

        with torch.no_grad():
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
        loss_theta = self.criterion(online_pred_theta, target_proj_xi.detach().clone())
        loss_xi = self.criterion(online_pred_xi, target_proj_theta.detach().clone())
        return (loss_theta + loss_xi) / 2

    def on_train_start(self, train_loader, epoch_num, *args, **kwargs):
        self.iter_num_in_epoch = len(train_loader)
        self.max_iter = self.iter_num_in_epoch * epoch_num

    def on_train_epoch_start(self, n_epoch):
        self.cur_iter = self.iter_num_in_epoch * (n_epoch - 1)

    def on_train_iter_end(self):
        # update momentum
        with torch.no_grad():
            self.m_encoder.update()
            self.m_projector.update()
        self.cur_iter += 1
        self.m_encoder.update_m_factor(self.cur_iter, self.max_iter)
        self.m_projector.update_m_factor(self.cur_iter, self.max_iter)

