from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.cuda import amp
from torch.nn import CrossEntropyLoss
import torchvision

from . import networks, dataset, optimizer, lr_scheduler, metrics, loss_fn
import utils


class FineTuner(object):
    def __init__(self, cfg, mlflow_logger, pretrain_mlflow_logger, tb_logger):
        # define base
        self.fine_tune_params = cfg.fine_tune_params
        self.mlflow_logger = mlflow_logger
        self.tb_logger = tb_logger
        self.start_epoch = 1
        self.best_score = 0.0

        # define gpu
        use_cuda = self.fine_tune_params.gpu_id > -1 and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{self.fine_tune_params.gpu_id}' if use_cuda else "cpu")

        # define loss
        criterion_class = getattr(eval(cfg.loss_params.root), cfg.loss_params.name)
        self.criterion = criterion_class(**cfg.loss_params.params)

        # define metrics
        self.metrics = metrics.get_metrics('accuracy')

        # define networks
        pretrain_model = pretrain_mlflow_logger.load_torch_model(self.fine_tune_params.model_state)
        self.model = networks.WrapperClassifier(pretrain_model.encoder, cfg.dataset.num_classes,
                                                fix_encoder=self.fine_tune_params.fix_encoder)
        self.model.to(self.device)  # model to device

        # define optimizer
        self.optimizer = optimizer.get_optimizer(self.model, cfg.optimizer)
        self.scheduler = lr_scheduler.get_lr_scheduler(self.optimizer, self.fine_tune_params.epoch_num, cfg.lr_sheduler)
        self.scaler = amp.GradScaler(enabled=self.fine_tune_params.use_amp)

        # load resume
        if self.fine_tune_params.resume.use:
            resume_mlflow_logger = utils.MlflowLogger('fine_tune', wd_root_dict=self.mlflow_logger.wd_root_dict)
            resume_mlflow_logger.set_run_id_from_run_name(self.fine_tune_params.resume.run_name)
            self.__load_checkpoints(resume_mlflow_logger, f'checkpoints_{self.fine_tune_params.resume.checkpoints}')
            utils.transfer_metrics_data(resume_mlflow_logger, self.mlflow_logger, self.fine_tune_params.resume.checkpoints)

        # define data_loader
        self.train_loader, self.val_loader, _ = dataset.get_dataloaders(
            self.fine_tune_params.batch_size, self.fine_tune_params.multi_cpu_num, cfg.dataset, cfg.aug_params)
        assert self.val_loader is not None, "The number of validation image must be more than 0"

        # # # log tensorboard
        # log batch sample
        self.mean, self.std = cfg.dataset.normalize.mean, cfg.dataset.normalize.std
        self.__log_batch_multi_image(self.train_loader, state='BatchSample/Train', is_multi_img=True)
        self.__log_batch_multi_image(self.val_loader, state='BatchSample/Validation', is_multi_img=False)

    def run(self):
        for n_epoch in range(self.start_epoch, self.fine_tune_params.epoch_num+1):
            # train
            train_metrics_dict = self.train(n_epoch)
            if n_epoch % self.fine_tune_params.log_interval.train == 0:
                self.mlflow_logger.log_metrics(train_metrics_dict, n_epoch)
                utils.log_metrics_for_tb(self.tb_logger, train_metrics_dict, n_epoch)

            # validation
            if n_epoch % self.fine_tune_params.log_interval.validation == 0:
                val_metrics_dict = self.validation(n_epoch)
                self.mlflow_logger.log_metrics(val_metrics_dict, n_epoch)
                utils.log_metrics_for_tb(self.tb_logger, val_metrics_dict, n_epoch)
                if val_metrics_dict['Accuracy/Validation'] > self.best_score:
                    self.__save_checkpoints('best', n_epoch)
                    self.best_score = val_metrics_dict['Accuracy/Validation']

            # save checkpoints
            if n_epoch % self.fine_tune_params.log_interval.checkpoints == 0:
                self.__save_checkpoints(f'checkpoints_{n_epoch}', n_epoch)

    def train(self, n_epoch):
        self.model.train()  # model reconfiguration

        # training iteration loop
        loss_logger, acc_logger = utils.AverageMeter(), utils.AverageMeter()
        with tqdm(self.train_loader, unit='batch', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as tepoch:
            tepoch.set_description(f"Train Epoch {n_epoch}")

            for image, teach, _ in tepoch:
                image = image[0].to(self.device, non_blocking=True)
                teach = teach.to(self.device, non_blocking=True)

                # forward
                with amp.autocast(enabled=self.fine_tune_params.use_amp):
                    logits = self.model(image)
                    loss = self.criterion(logits, teach)
                    soft_label = F.softmax(logits, dim=1)

                # update parameter
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # logging
                loss_logger.update(loss.item(), image.size(0))
                prec1 = self.metrics(soft_label, teach)[0]
                acc_logger.update(prec1.item(), image.size(0))
                tepoch.set_postfix(avg_acc=acc_logger.avg)  # tqdm
        self.scheduler.step()

        # summarize log data
        train_metrics_dict = {
            'Loss/Train': loss_logger.avg,
            'Accuracy/Train': acc_logger.avg,
            'LearningRate/Train': self.scheduler.get_last_lr()[0]
        }
        return train_metrics_dict

    def validation(self, n_epoch):
        self.model.eval()  # model reconfiguration

        # validation iteration loop
        loss_logger, acc_logger = utils.AverageMeter(), utils.AverageMeter()
        with tqdm(self.val_loader, unit='batch', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as tepoch:
            tepoch.set_description(f"Validation Epoch {n_epoch}")

            for image, teach, _ in tepoch:
                image = image.to(self.device, non_blocking=True)
                teach = teach.to(self.device, non_blocking=True)

                with torch.no_grad():
                    logits = self.model(image)
                    loss = self.criterion(logits, teach)
                    soft_label = F.softmax(logits, dim=1)

                # logging
                loss_logger.update(loss.item(), image.size(0))
                prec1 = self.metrics(soft_label, teach)[0]
                acc_logger.update(prec1.item(), image.size(0))
                tepoch.set_postfix(avg_acc=acc_logger.avg)  # tqdm

        val_metrics_dict = {
            'Loss/Validation': loss_logger.avg,
            'Accuracy/Validation': acc_logger.avg
        }

        return val_metrics_dict

    def __load_checkpoints(self, mlflow_logger, state):
        model = mlflow_logger.load_torch_model(state)

        checkpoint = mlflow_logger.load_torch_state_dict(state)
        self.start_epoch = checkpoint['n_epoch'] + 1
        self.model.load_state_dict(model.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_score = checkpoint['best_score']

    def __save_checkpoints(self, state, n_epoch):
        self.mlflow_logger.log_torch_model(self.model, state)

        check_point = {
            'n_epoch': n_epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_score': self.best_score
        }
        self.mlflow_logger.log_torch_state_dict(check_point, state)

    def __log_batch_multi_image(self, data_loader, state, is_multi_img=False):
        img_list = iter(data_loader).__next__()[0]
        img_list = img_list if is_multi_img else [img_list]
        for idx, img in enumerate(img_list):
            grid_img = utils.get_batch_sample_img(img, self.mean, self.std, resize_ratio=1., log_img_num=32)
            self.tb_logger.add_image(f'{state}{idx}', grid_img, 0)

