import torch
from torch.optim.lr_scheduler import StepLR
from . import cosine_scheduler


def get_lr_scheduler(optimizer, training_steps, cfg_lr_schedule):

    if cfg_lr_schedule.name == 'CosineScheduler':
        scheduler_class = getattr(eval(cfg_lr_schedule.root), cfg_lr_schedule.name)
        scheduler = scheduler_class(optimizer, training_steps, **cfg_lr_schedule.params)

    elif cfg_lr_schedule.name == 'None':
        scheduler = StepLR(optimizer, step_size=training_steps, gamma=1.0)  # 学習率に変化なし

    elif cfg_lr_schedule.name in ['CosineAnnealingWarmupRestarts', 'MultiStepLR']:
        scheduler_class = getattr(eval(cfg_lr_schedule.root), cfg_lr_schedule.name)
        scheduler = scheduler_class(optimizer, **cfg_lr_schedule.params)

    else:
        raise NotImplementedError(f'Not implemented lr_schedule: {cfg_lr_schedule.name}')

    return scheduler

