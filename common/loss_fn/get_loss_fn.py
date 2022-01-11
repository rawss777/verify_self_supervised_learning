import torch
from . import info_nce_loss, cross_entropy_loss, cosine_similarity_loss


def get_ssl_loss_fn(n_views, device, cfg_loss):
    loss_class = getattr(eval(cfg_loss.root), cfg_loss.name)
    params = {} if cfg_loss.params is None else cfg_loss.params
    loss_fn = loss_class(n_views, device, **params)

    return loss_fn


def get_cls_loss_fn(cfg_loss):
    loss_class = getattr(eval(cfg_loss.root), cfg_loss.name)
    params = {} if cfg_loss.params is None else cfg_loss.params
    loss_fn = loss_class(**params)

    return loss_fn

