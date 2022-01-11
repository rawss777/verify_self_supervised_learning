import os
import warnings

import hydra  # need install hydra
from omegaconf import DictConfig  # need install omegaconf
import mlflow
from torch.utils.tensorboard import SummaryWriter

import utils
from common import Trainer


@hydra.main(config_path='params', config_name='train')
def main(cfg: DictConfig):
    if cfg.experiment_params.debug_mode:
        warnings.warn('This training is DEBUG MODE')

    # # # setting dir root
    wd_root_dict = {}
    wd_root_dict['current'] = hydra.utils.get_original_cwd()                                  # current root
    wd_root_dict['hydra'] = os.getcwd()                                                       # hydra save root
    wd_root_dict['output'] = f'{wd_root_dict["current"]}/{cfg.experiment_params.output_dir}'  # mlflow save root
    os.chdir(wd_root_dict['current'])

    # # # mlflow logger setting
    # set experiment_id
    mlflow_logger = utils.MlflowLogger(experiment_name='train', wd_root_dict=wd_root_dict)
    # set run_id
    mlflow_logger.create_run(cfg.experiment_params.run_name, cfg.experiment_params.debug_mode)
    mlflow_logger.log_tag(key="run_id", value=mlflow_logger.run_id)
    print(f'Experiment [Name: {mlflow_logger.experiment_name}, Id: {mlflow_logger.experiment_id}]')
    print(f'Run        [Name: {mlflow_logger.run_name}, Id: {mlflow_logger.run_id}]')
    # save params info to mlflow result
    mlflow_logger.log_params_from_omegaconf_dict(cfg, save_dir='train_hydra_config')

    # tensorboard logger setting
    tb_root = f'{wd_root_dict["output"]}/tensorboard/{mlflow_logger.experiment_name}/{mlflow_logger.run_name}'
    tb_logger = SummaryWriter(tb_root)

    # set seed
    utils.seed_settings(cfg.experiment_params.seed)  # seedがNoneの場合，ランダムシード

    # select trainer
    trainer = Trainer(cfg, mlflow_logger, tb_logger)

    # # run
    trainer.run()
    tb_logger.close()


if __name__ == '__main__':
    main()
