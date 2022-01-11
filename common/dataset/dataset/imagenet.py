import numpy as np
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def get_imagenet(cfg_dataset):

def get_cifar10(cfg_dataset):
    base_dataset = datasets.CIFAR10(cfg_dataset.dataset_path, train=True, download=cfg_dataset.download)
    assert (cfg_dataset.num_train % cfg_dataset.num_classes == 0) and \
           (0 < cfg_dataset.num_train) and \
           (cfg_dataset.num_train <= len(base_dataset)), \
        f"The number of training image ({cfg_dataset.num_train}) must be " \
        f"less than {len(base_dataset)} and more than 0"

    if cfg_dataset.num_train == len(base_dataset):
        train_dataset = DatasetCIFAR10(cfg_dataset.dataset_path, None, train=True)
        val_dataset = None
        test_dataset = DatasetCIFAR10(cfg_dataset.dataset_path, None, train=False, download=cfg_dataset.download)
    else:
        train_idx_list, val_idx_list = make_split_idx_list(cfg_dataset, base_dataset.targets)
        train_dataset = DatasetCIFAR10(cfg_dataset.dataset_path, train_idx_list, train=True)
        val_dataset = DatasetCIFAR10(cfg_dataset.dataset_path, val_idx_list, train=True)
        test_dataset = DatasetCIFAR10(cfg_dataset.dataset_path, None, train=False, download=cfg_dataset.download)

    return train_dataset, val_dataset, test_dataset


def make_split_idx_list(cfg_dataset, labels):
    labels = np.array(labels)
    train_per_class = cfg_dataset.num_train // cfg_dataset.num_classes

    train_idx_list, val_idx_list = [], []
    for idx in range(cfg_dataset.num_classes):
        target_class_idx = np.where(labels == idx)[0]

        # extract validation idx
        train_idx = np.random.choice(target_class_idx, train_per_class, False)
        val_idx = np.delete(target_class_idx, np.isin(target_class_idx, train_idx))
        train_idx_list.append(train_idx)
        val_idx_list.append(val_idx)

    train_idx_list = np.concatenate(train_idx_list, axis=0)
    val_idx_list = np.concatenate(val_idx_list, axis=0)
    return train_idx_list, val_idx_list


class DatasetCIFAR10(datasets.CIFAR10):
    def __init__(self, root, index, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if index is not None:
            self.data = self.data[index]
            self.targets = np.array(self.targets)[index]

    def __getitem__(self, i):
        img, target = self.data[i], self.targets[i]
        img = Image.fromarray(img.astype(np.uint8))
        img = self.transform(img)  # albumentations = Numpy array, torchvision = PIL

        return img, target

