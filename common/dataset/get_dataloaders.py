from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from . import augmentations
from . import dataset


def get_dataloaders(batch_size, multi_cpu_num, cfg_dataset, cfg_aug):
    # get transform
    train_transform, test_transform = augmentations.get_transforms(cfg_dataset, cfg_aug)

    # get dataset
    if cfg_dataset.name == 'cifar10':
        train_dataset, val_dataset, test_dataset = dataset.get_cifar10(cfg_dataset)
    elif cfg_dataset.name == 'stl10':
        train_dataset, val_dataset, test_dataset = dataset.get_stl10(cfg_dataset)
    else:
        raise NotImplementedError(f'Not implemented dataset: {dataset_name}')

    # # # set dataloader
    # train
    train_dataset.transform = train_transform
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                  batch_size=batch_size, num_workers=multi_cpu_num, drop_last=True)
    # test
    test_dataset.transform = test_transform
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                 batch_size=batch_size, num_workers=multi_cpu_num)
    # validation
    val_dataloader = None
    if val_dataset is not None:
        val_dataset.transform = test_transform
        val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset),
                                    batch_size=batch_size, num_workers=multi_cpu_num)

    return train_dataloader, val_dataloader, test_dataloader
