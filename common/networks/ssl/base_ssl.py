import torch
from torch import nn

from ... import loss_fn
from .util_modules import SplitBatchNorm1d, SplitBatchNorm2d


class BaseSSL(nn.Module):
    def __init__(self, encoder: nn.Module, device: str, loss_params: dict):
        super(BaseSSL, self).__init__()
        self.device = device
        self.encoder = encoder
        self.criterion = self.get_criterion_fn(loss_params)

    @staticmethod
    def get_criterion_fn(loss_params):
        criterion_class = getattr(eval(loss_params["root"]), loss_params["name"])
        params = {} if loss_params["params"] is None else loss_params["params"]
        criterion_fn = criterion_class(**params)
        return criterion_fn

    def forward(self, multi_img_list: list, *args, **kwargs):
        raise NotImplementedError('Must define "forward" method')

    def on_train_start(self, *args, **kwargs):
        pass

    def on_train_epoch_start(self, *args, **kwargs):
        pass

    def on_train_iter_end(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def replace_bn_to_split_bn(self, modules: torch.nn.Module, num_split: int):
        def __replace(replace_mod, mod_root, ori_modules):
            split_list = mod_root.split('.')
            mod_name = split_list[-1]
            num_root = len(split_list) - 1
            if num_root == 0:
                setattr(ori_modules, mod_name, replace_mod)
            elif num_root == 1:
                pre_module = getattr(ori_modules, split_list[0])
                setattr(pre_module, mod_name, replace_mod)
            else:
                pre_module = getattr(ori_modules, split_list[0])
                for s in split_list[1:-1]:
                    pre_module = getattr(pre_module, s)
                setattr(pre_module, mod_name, replace_mod)

        for name, module in modules.named_modules():
            is_replace = False
            if isinstance(module, nn.BatchNorm2d):
                split_bn = SplitBatchNorm2d(module.num_features, num_split)
                is_replace = True
            elif isinstance(module, nn.BatchNorm1d):
                split_bn = SplitBatchNorm1d(module.num_features, num_split)
                is_replace = True

            if is_replace:
                __replace(split_bn, name, modules)

    @torch.no_grad()
    def batch_shuffle(self, img):
        batch_size = img.shape[0]
        shuffle_idx = torch.randperm(batch_size).to(img.device)
        shuffled_img = img[shuffle_idx]
        unshuffle_idx = torch.argsort(shuffle_idx)
        return shuffled_img, unshuffle_idx

    @torch.no_grad()
    def batch_revert(self, shuffled_img, unshuffle_idx):
        return shuffled_img[unshuffle_idx]

