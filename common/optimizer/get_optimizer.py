import torch
from . import lars


def get_optimizer(model, cfg_optimizer):
    params = lars.__configure_params(model) if "LARS" == cfg_optimizer.name else model.parameters()
    opt_class = getattr(eval(cfg_optimizer.root), cfg_optimizer.name)
    optimizer = opt_class(params, **cfg_optimizer.params)

    return optimizer



