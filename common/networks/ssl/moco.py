import torch
from torch import nn

from .base_ssl import BaseSSL
from .util_modules import MLP, EMA, EMAN, QueueMemory


class MoCo(BaseSSL):
    def __init__(self, encoder: nn.Module, criterion: nn.Module, device: str,
                 proj_params: dict, m_factor: float, memory_queue_params: dict, split_bn: dict, use_eman: bool):
        # define query encoder, criterion, device
        super().__init__(encoder, criterion, device)

        # define query networks
        self.projector = MLP(input_dim=encoder.output_dim, **proj_params)

        # define key networks
        ema_mod = EMAN if use_eman else EMA
        self.m_encoder = ema_mod(self.encoder, m_factor)
        self.m_projector = ema_mod(self.projector, m_factor)

        # replace key networks to split_bn
        self.use_split_bn = split_bn.use
        if self.use_split_bn:
            self.replace_bn_to_split_bn(self.m_encoder, split_bn.num_split)
            self.replace_bn_to_split_bn(self.m_projector, split_bn.num_split)

        # define memory queue
        self.memory_bank = QueueMemory(memory_queue_params["memory_size"])
        self.dequeue_size = memory_queue_params["dequeue_size"]

    def forward(self, multi_img_list: list, *args, **kwargs):
        xq, xk = multi_img_list[0], multi_img_list[1]
        xq = xq.to(self.device, non_blocking=True)
        xk = xk.to(self.device, non_blocking=True)

        # get query projection
        proj_q = self.projector(self.encoder(xq))

        with torch.no_grad():
            # update momentum
            self.m_encoder.update()
            self.m_projector.update()

            # batch shuffle
            if self.use_split_bn:
                xk, unshuffled_idx = self.batch_shuffle(xk)

            # get key projection
            proj_k = self.m_projector(self.m_encoder(xk))

            # batch revert
            if self.use_split_bn:
                proj_k = self.batch_revert(proj_k, unshuffled_idx)

        # enqueue and dequeue negative sample
        proj_only_neg = None
        if len(self.memory_bank) > self.dequeue_size:
            proj_only_neg = self.memory_bank.dequeue(self.dequeue_size)
        self.memory_bank.enqueue(proj_k.detach())

        # compute loss
        proj = torch.cat([proj_q, proj_k], dim=0)
        loss = self.criterion(proj, negative_feature=proj_only_neg)
        return loss

