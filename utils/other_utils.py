import os, glob
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid


# # # for tensorboard # # #
def get_batch_sample_img(img: torch.Tensor, mean: list, std: list, resize_ratio=1., log_img_num=32):
    log_img_num = min(log_img_num, len(img))
    img = img[:log_img_num]

    # back to original image and resize for logging
    for i in range(img.shape[1]):
        img[:, i] = (img[:, i] * std[i] + mean[i]) * 255
    img = F.interpolate(img, scale_factor=resize_ratio, mode='bicubic', align_corners=True)
    img = img.clamp(min=0, max=255)

    # make grid img
    row_num = min(8, len(img))
    grid_img = make_grid(img, nrow=row_num).to(torch.uint8)
    return grid_img


def log_metrics_for_tb(tb_logger, metrics_dict: dict, n_iter: int):
    for key, value in metrics_dict.items():
        tb_logger.add_scalar(key, value, n_iter)
# # # for tensorboard # # #


# # # for mlflow # # #
def transfer_metrics_data(from_mlflow_logger, to_mlflow_logger, resume_n_epoch: int):
    # get FROM metrics root
    from_artifact_root = from_mlflow_logger.get_artifact_root()
    root_split_list = from_artifact_root.split('/')
    root_split_list[-1] = 'metrics'
    from_metrics_root = '/'.join(root_split_list)

    # transfer metrics data
    metrics_file_list = [p for p in glob.glob(f'{from_metrics_root}/**', recursive=True) if os.path.isfile(p)]
    for metrics_file in metrics_file_list:
        metrics_data = np.loadtxt(metrics_file, dtype=str, delimiter=' ')
        key = metrics_file.replace(f'{from_metrics_root}/', '')
        for _, value, n_epoch in metrics_data:
            if int(n_epoch) <= resume_n_epoch:
                to_mlflow_logger.log_metrics({key: float(value)}, int(n_epoch))
# # # for mlflow # # #

