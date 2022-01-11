import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ResultStacker(object):
    def __init__(self):
        self.result_list = []
        self.header = ['teach', 'predict', 'certainty', 'accurate']

    def reset(self):
        self.result_list = []

    def update(self, teach, predict, certainty, accurate):
        self.result_list.extend([[t, p, round(c, 2), a] for t, p, c, a in zip(teach, predict, certainty, accurate)])

    def save(self, save_path):
        df = pd.DataFrame(self.result_list, columns=self.header, index=None)
        df.to_csv(save_path, index=None)


class FeatureStacker(object):
    def __init__(self, module, device):
        self.use_gpu = True if device.type == 'cuda' else False
        self.feature_list = []
        self.handle = module.register_forward_hook(self.__stack)

    def __stack(self, module, module_in, module_out):
        feature = module_out.detach().clone()
        if self.use_gpu: feature.cpu()
        self.feature_list.append(feature)

    def get_feature(self):
        feature_list = torch.cat(self.feature_list, dim=0)
        return feature_list

    def release_register_forward_hook(self):
        self.handle.remove()


class ImageStacker(object):
    def __init__(self, resize_ratio, mean, std, device):
        self.img_list = []
        self.use_gpu = True if device.type == 'cuda' else False
        self.mean = torch.Tensor(mean).unsqueeze(1).unsqueeze(2).to(device)
        self.std = torch.Tensor(std).unsqueeze(1).unsqueeze(2).to(device)
        self.resize_ratio = resize_ratio

    def stack(self, img):
        img = img.detach() * self.std + self.mean
        img = F.interpolate(img, scale_factor=self.resize_ratio)
        if self.use_gpu: img.cpu()
        self.img_list.append(img)

    def get_image(self):
        img_list = torch.cat(self.img_list, dim=0)
        img_list = torch.clip(img_list, min=0.0, max=1.0)
        return img_list

