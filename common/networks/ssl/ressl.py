import torch
from torch import nn

from .base_ssl import BaseSSL
from .util_modules import MLP, EMA, EMAN, QueueMemory


class ReSSL(BaseSSL):
    def __init__(self, encoder: nn.Module, device: str, loss_params: dict,
                 proj_params: dict, m_factor: dict, memory_queue_params: dict, use_eman: bool):
        # define query encoder, device, criterion
        super().__init__(encoder, device, loss_params)

        # define student networks
        self.projector = MLP(input_dim=encoder.output_dim, **proj_params)

        # define teacher networks
        ema_mod = EMAN if use_eman else EMA
        self.m_encoder = ema_mod(self.encoder, **m_factor)
        self.m_projector = ema_mod(self.projector, **m_factor)

        # define memory queue
        self.memory_bank = QueueMemory(memory_queue_params["memory_size"])
        self.dequeue_size = memory_queue_params["dequeue_size"]

        # define in on_train_start
        self.max_iter = None
        self.iter_num_in_epoch = None
        # define in on_train_epoch_start
        self.cur_iter = 0

    def forward(self, multi_img_list: list, *args, **kwargs):
        xq, xk = multi_img_list[0], multi_img_list[1]
        xq = xq.to(self.device, non_blocking=True)
        xk = xk.to(self.device, non_blocking=True)

        # get query projection
        proj_q = self.projector(self.encoder(xq))

        # get key projection
        with torch.no_grad():
            proj_k = self.m_projector(self.m_encoder(xk))

        # compute loss
        loss = 0.
        if len(self.memory_bank) > self.dequeue_size:
            queue = self.memory_bank.dequeue(self.dequeue_size)
            loss = self.criterion(proj_q, proj_k, queue)
        self.memory_bank.enqueue(proj_k.detach().clone())

        return loss

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


