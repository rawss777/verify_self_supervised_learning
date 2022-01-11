import torchvision
from torchvision import transforms
from . import util_augments


def get_transforms(cfg_dataset, cfg_aug):
    # train transform
    train_transform = MultiViewTransform(cfg_dataset, cfg_aug)

    # validation and test transform
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg_dataset.normalize.mean, std=cfg_dataset.normalize.std),
    ])

    return train_transform, test_transform


class MultiViewTransform(object):
    def __init__(self, cfg_dataset, cfg_aug):
        self.mean = cfg_dataset.normalize.mean
        self.std = cfg_dataset.normalize.std

        self.view_list = []
        for view_name in cfg_aug.multi_view_list:
            transform_dict = cfg_aug.views[view_name]
            single_transform = self.__get_single_view_transform(transform_dict)
            self.view_list.append(single_transform)

    def __get_single_view_transform(self, view_dict):
        transform_list = []
        for aug in view_dict.values():
            if aug.use:
                aug_class = getattr(eval(aug.root), aug.name)
                transform_list.append(aug_class(**aug.params))

        # post process
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        return transforms.Compose(transform_list)

    def __call__(self, img):
        return tuple([op(img) for op in self.view_list])

