import numpy as np
from tqdm import tqdm
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf

from . import metrics
from . import dataset
import utils


class Tester(object):
    def __init__(self, cfg, mlflow_logger, train_mlflow_logger, tb_logger):
        self.test_params = cfg.test_params
        self.mlflow_logger = mlflow_logger
        self.tb_logger = tb_logger

        # define gpu
        use_cuda = self.test_params.gpu_id > -1 and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{self.test_params.gpu_id}' if use_cuda else "cpu")

        # define networks
        self.model = train_mlflow_logger.load_torch_model(self.test_params.model_state)
        self.model.eval()

        # define metrics
        self.metrics = metrics.get_metrics('accuracy')

        # define data_loader
        train_hydra_config_root = f'{train_mlflow_logger.get_artifact_root()}/fine_tune_hydra_config'
        train_config = OmegaConf.load(f'{train_hydra_config_root}/.hydra/config.yaml')
        _, _, self.test_loader = dataset.get_dataloaders(
            self.test_params.batch_size, self.test_params.multi_cpu_num, train_config.dataset, train_config.augmentation)

        # save train config to test run
        self.mlflow_logger.log_artifacts(train_hydra_config_root, save_dir='fine_tune_hydra_config')

        # # # log tensorboard
        # log batch sample
        self.mean, self.std = train_config.dataset.normalize.mean, train_config.dataset.normalize.std
        self.__log_batch_multi_image(self.test_loader, state='BatchSample/Test', is_multi_img=False)

    def run(self):
        acc_stacker = utils.AverageMeter()
        result_stacker = utils.ResultStacker()
        feature_stacker = utils.FeatureStacker(self.model.encoder, self.device)
        # image_stacker = utils.ImageStacker(1.0, self.mean, self.std, self.device)

        with tqdm(self.test_loader, unit='batch', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as tepoch:
            tepoch.set_description(f"Test ")

            for image, teach, _ in tepoch:
                image = image.to(self.device, non_blocking=True)
                teach = teach.to(self.device, non_blocking=True)

                with torch.no_grad():
                    logits = self.model(image)
                    soft_label = F.softmax(logits, dim=1)
                max_soft_label, predict_label = soft_label.max(dim=1)

                prec1 = self.metrics(soft_label, teach, topk=(1,))[0]
                acc_stacker.update(prec1.item(), image.size(0))
                tepoch.set_postfix(avg_accuracy=acc_stacker.avg)

                teach, predict_label = teach.cpu().int(), predict_label.cpu().int()
                max_soft_label = max_soft_label.cpu()
                accurate = (teach == predict_label).int()
                result_stacker.update(teach.tolist(), predict_label.tolist(), max_soft_label.tolist(), accurate.tolist())
                # image_stacker.stack(image)

        # mlflow logging
        save_csv_path = f'{self.mlflow_logger.get_artifact_root()}/predict.csv'
        result_stacker.save(save_csv_path)
        self.mlflow_logger.log_metrics({'top1_acc': acc_stacker.avg})

        # tensorboard logging
        feature = feature_stacker.get_feature()
        self.tb_logger.add_embedding(feature, metadata=result_stacker.result_list,
                                     metadata_header=result_stacker.header, tag=f'HiddenFeature')
        # feature, img = feature_stacker.get_feature(), image_stacker.get_image()
        # self.tb_logger.add_embedding(feature, metadata=result_stacker.result_list, label_img=img,
        #                              metadata_header=result_stacker.header, tag='hidden_feature')

    def __log_batch_multi_image(self, data_loader, state, is_multi_img=False):
        img_list = iter(data_loader).__next__()[0]
        img_list = img_list if is_multi_img else [img_list]
        for idx, img in enumerate(img_list):
            grid_img = utils.get_batch_sample_img(img, self.mean, self.std, resize_ratio=1., log_img_num=32)
            self.tb_logger.add_image(f'{state}{idx}', grid_img, 0)
