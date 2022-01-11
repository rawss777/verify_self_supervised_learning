from .accuracy import accuracy


def get_metrics(metrics_name='accuracy'):
    if metrics_name == 'accuracy':
        metrics = accuracy
    else:
        raise NotImplementedError(f'Not implemented lr_schedule: {cfg_lr_schedule.name}')

    return metrics

