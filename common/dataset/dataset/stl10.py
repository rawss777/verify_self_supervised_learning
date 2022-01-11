from copy import deepcopy
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def get_stl10(cfg_dataset):
    if cfg_dataset.unlabeled_to_train:
        train_dataset = DatasetSTL10(cfg_dataset.dataset_path, target_type='unlabeled')
        val_dataset = DatasetSTL10(cfg_dataset.dataset_path, target_type='train')
    else:
        train_dataset = DatasetSTL10(cfg_dataset.dataset_path, target_type='train')
        train_idx_list, val_idx_list = make_split_idx_list(cfg_dataset.num_classes, train_dataset.labels)
        val_dataset = deepcopy(train_dataset)
        train_dataset.extract_data_from_idx_list(train_idx_list)
        val_dataset.extract_data_from_idx_list(val_idx_list)

    test_dataset = DatasetSTL10(cfg_dataset.dataset_path, target_type='test')

    return train_dataset, val_dataset, test_dataset


def make_split_idx_list(num_classes, labels):
    train_per_class = int(len(labels) * 0.9) // num_classes  # num_validation = num_train * 0.9

    train_idx_list, val_idx_list = [], []
    for idx in range(num_classes):
        target_class_idx = np.where(labels == idx)[0]

        # extract validation idx
        train_idx = np.random.choice(target_class_idx, train_per_class, False)
        val_idx = np.delete(target_class_idx, np.isin(target_class_idx, train_idx))
        train_idx_list.append(train_idx)
        val_idx_list.append(val_idx)

    train_idx_list = np.concatenate(train_idx_list, axis=0)
    val_idx_list = np.concatenate(val_idx_list, axis=0)
    return train_idx_list, val_idx_list


class DatasetSTL10(Dataset):
    def __init__(self, root, target_type='train', transform=None, target_transform=None):
        super().__init__()
        self.transform = transform

        assert target_type in ["train", "test", "unlabeled"]
        image_path = f"{root}/stl10_binary/{target_type}_X.bin"
        label_path = f"{root}/stl10_binary/{target_type}_y.bin" if target_type != "unlabeled" else None
        self.images, self.labels = self.__load_data(image_path, label_path)

    def __getitem__(self, i):
        img, target = self.images[i], self.labels[i]
        img = Image.fromarray(img.astype(np.uint8))
        img = self.transform(img)  # albumentations = Numpy array, torchvision = PIL

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def __load_data(image_path: str, label_path: str):
        # image
        with open(image_path, "rb") as fp:
            images = np.fromfile(fp, dtype=np.uint8)
            images = images.reshape(-1, 3, 96, 96)
        images = np.transpose(images, (0, 3, 2, 1))  # to PIL shape

        # label
        if label_path is not None:
            with open(label_path) as fp:
                labels = np.fromfile(fp, dtype=np.uint8)
            labels = labels.reshape(-1,) - 1  # 1-10 -> 0-9
        else:
            labels = np.ones(images.shape[0],) * -1  # -1 (invalid value)

        return images, labels

    def extract_data_from_idx_list(self, idx_list):
        self.images = self.images[idx_list]
        self.labels = self.labels[idx_list]

