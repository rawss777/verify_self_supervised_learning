from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.cuda import amp
from torch.nn import CrossEntropyLoss
import torchvision

from . import networks, dataset, optimizer, lr_scheduler, metrics, loss_fn
import utils


class Trainer(object):
    def __init__(self, cfg, mlflow_logger, tb_logger):
        # define base
        self.train_params = cfg.train_params
        self.mlflow_logger = mlflow_logger
        self.tb_logger = tb_logger
        self.start_epoch = 1

        # define gpu
        use_cuda = self.train_params.gpu_id > -1 and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{self.train_params.gpu_id}' if use_cuda else "cpu")

        # define networks
        self.model = networks.get_network(self.device, cfg.encoder, cfg.self_supervised, cfg.dataset)
        self.model.to(self.device)  # model to device

        # define optimizer
        self.optimizer = optimizer.get_optimizer(self.model, cfg.optimizer)
        self.scheduler = lr_scheduler.get_lr_scheduler(self.optimizer, self.train_params.epoch_num, cfg.lr_sheduler)
        self.scaler = amp.GradScaler(enabled=self.train_params.use_amp)

        # define data_loader
        self.train_loader, self.val_loader, _ = dataset.get_dataloaders(
            self.train_params.batch_size, self.train_params.multi_cpu_num, cfg.dataset, cfg.self_supervised.aug_params)
        assert self.val_loader is not None, "The number of validation image must be more than 0"

        # log batch sample
        run_name = cfg.experiment_params.run_name
        self.mean, self.std = cfg.dataset.normalize.mean, cfg.dataset.normalize.std
        self.__log_batch_multi_image(self.train_loader, state=f'{run_name}/train', is_multi_img=True)
        self.__log_batch_multi_image(self.val_loader, state=f'{run_name}/validation', is_multi_img=False)

        # load resume
        if self.train_params.resume.use:
            resume_mlflow_logger = utils.MlflowLogger('train', wd_root_dict=self.mlflow_logger.wd_root_dict)
            resume_mlflow_logger.set_run_id_from_run_name(self.train_params.resume.run_name)
            self.__load_checkpoints(resume_mlflow_logger, f'checkpoints_{self.train_params.resume.checkpoints}')
            utils.transfer_metrics_data(resume_mlflow_logger, self.mlflow_logger, self.train_params.resume.checkpoints)

    def run(self):
        self.on_train_start()  # before training process

        for n_epoch in range(self.start_epoch, self.train_params.epoch_num+1):
            # train
            self.on_train_epoch_start(n_epoch)  # before training process each epochs
            train_metrics_dict = self.train(n_epoch)
            if n_epoch % self.train_params.log_interval.train == 0:
                self.mlflow_logger.log_metrics(train_metrics_dict, n_epoch)
                utils.log_metrics_for_tb(self.tb_logger, train_metrics_dict, n_epoch)

            # validation
            if n_epoch % self.train_params.log_interval.validation == 0:
                self.validation(n_epoch)

            # save checkpoints
            if n_epoch % self.train_params.log_interval.checkpoints == 0:
                self.__save_checkpoints(f'checkpoints_{n_epoch}', n_epoch)

        # save last
        self.__save_checkpoints('last', self.train_params.epoch_num)

    def train(self, n_epoch):
        self.model.train()  # model reconfiguration

        # training iteration loop
        loss_logger = utils.AverageMeter()
        with tqdm(self.train_loader, unit='batch', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as tepoch:
            tepoch.set_description(f"Train Epoch {n_epoch}")

            for idx, (multi_img_list, _, idx_list) in enumerate(tepoch):
                # forward
                with amp.autocast(enabled=self.train_params.use_amp):
                    loss = self.model(multi_img_list, idx_list)

                # update parameter
                if (loss is not None) and (torch.is_tensor(loss)):
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    v_loss = loss.item()
                else:
                    v_loss = 0.

                # logging
                batch_size = multi_img_list[0].shape[0]
                loss_logger.update(v_loss, batch_size)
                tepoch.set_postfix(avg_loss=loss_logger.avg)  # tqdm

                # post process
                self.on_train_iter_end()
        self.scheduler.step()

        # summarize log data
        train_metrics_dict = {
            'Loss/Train': loss_logger.avg,
            'LearningRate/Train': self.scheduler.get_last_lr()[0]
        }
        return train_metrics_dict

    def validation(self, n_epoch):
        self.model.eval()  # model reconfiguration
        teach_list = []
        feature_stacker = utils.FeatureStacker(self.model.encoder, self.device)

        # validation iteration loop
        with tqdm(self.val_loader, unit='batch', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as tepoch:
            tepoch.set_description(f"Validation Epoch {n_epoch}")

            for image, teach, _ in tepoch:
                image = image.to(self.device, non_blocking=True)
                with torch.no_grad():
                    _ = self.model.encoder(image)
                teach_list.append(teach)

        # tensorboard logging
        teach_list = torch.cat(teach_list, dim=0).unsqueeze(-1).tolist()
        feature = feature_stacker.get_feature()
        self.tb_logger.add_embedding(feature, metadata=teach_list,
                                     metadata_header=None, tag=f'HiddenFeature/Epoch{n_epoch}')
        feature_stacker.release_register_forward_hook()
        del feature_stacker

    def on_train_start(self):
        self.model.on_train_start(self.train_loader, self.train_params.epoch_num)

    def on_train_epoch_start(self, n_epoch):
        self.model.on_train_epoch_start(n_epoch)

    def on_train_iter_end(self):
        self.model.on_train_iter_end()

    def __load_checkpoints(self, mlflow_logger, state):
        model = mlflow_logger.load_torch_model(state)

        checkpoint = mlflow_logger.load_torch_state_dict(state)
        self.start_epoch = checkpoint['n_epoch'] + 1
        self.model.load_state_dict(model.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

    def __save_checkpoints(self, state, n_epoch):
        self.mlflow_logger.log_torch_model(self.model, state)

        check_point = {
            'n_epoch': n_epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
        }
        self.mlflow_logger.log_torch_state_dict(check_point, state)

    def __log_batch_multi_image(self, data_loader, state, is_multi_img=False):
        img_list = iter(data_loader).__next__()[0]
        img_list = img_list if is_multi_img else [img_list]
        for idx, img in enumerate(img_list):
            grid_img = utils.get_batch_sample_img(img, self.mean, self.std, resize_ratio=1., log_img_num=32)
            self.tb_logger.add_image(f'{state}{idx}', grid_img, 0)

