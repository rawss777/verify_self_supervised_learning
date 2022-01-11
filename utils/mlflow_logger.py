import os
import logging

import hydra
import mlflow
from mlflow import pytorch
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, ListConfig
from mlflow.utils import requirements_utils
requirements_utils._logger.setLevel(logging.ERROR)


class MlflowLogger(object):
    def __init__(self, experiment_name, wd_root_dict, **kwargs):
        self.wd_root_dict = wd_root_dict
        self.client = MlflowClient(f"{self.wd_root_dict['output']}/mlruns", **kwargs)  # make client instance
        self.experiment_name = experiment_name

        # set experiment_id (既にexperiment_nameが存在するなら，そのexperiment_idを取得する)
        try:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id
        except:
            self.experiment_id = self.client.create_experiment(experiment_name)

    # set run methods
    def create_run(self, run_name, debug_mode=False):
        # 既に使われているrun_nameか判定する
        run_list = self.client.search_runs(experiment_ids=self.experiment_id,
                                           filter_string=f"tags.mlflow.runName = '{run_name}'")
        if len(run_list) == 0:
            self.run_id = self.client.create_run(self.experiment_id).info.run_id    # run_idの発行
        else:
            if debug_mode:
                self.set_run_id_from_run_name(run_name)
            else:
                raise ValueError(f'Already exist run_name: {run_name}, run_id: {run_list[0].info.run_id}')  # 使われていればエラー

        self.run_name = run_name
        self.client.set_tag(self.run_id, key="mlflow.runName", value=run_name)  # run_idにrun_nameを設定

    def set_run_id_from_run_name(self, run_name):
        # run_nameからrun_idを設定する
        run_list = self.client.search_runs(experiment_ids=self.experiment_id,
                                           filter_string=f"tags.mlflow.runName = '{run_name}'")
        if len(run_list) != 1:
            raise ValueError(f'Not exist run_name: {run_name}')

        self.run_name = run_name
        self.run_id = run_list[0].info.run_id                                   # run_idの取得

    # model (To artifacts directory)
    def load_torch_model(self, target_dir):
        os.chdir(self.wd_root_dict['output'])
        with mlflow.start_run(run_id=self.run_id) as run:
            state_dict_uri = f"runs:/{run.info.run_id}/{target_dir}"
            state_dict = pytorch.load_model(state_dict_uri)
        os.chdir(self.wd_root_dict['current'])
        return state_dict

    def log_torch_model(self, model, save_dir):
        os.chdir(self.wd_root_dict['output'])
        with mlflow.start_run(run_id=self.run_id):
            pytorch.log_model(model, artifact_path=save_dir)
        os.chdir(self.wd_root_dict['current'])

    # state_dict (To artifacts directory)
    def load_torch_state_dict(self, target_dir):
        os.chdir(self.wd_root_dict['output'])
        with mlflow.start_run(run_id=self.run_id) as run:
            state_dict_uri = f"runs:/{run.info.run_id}/{target_dir}"
            state_dict = pytorch.load_state_dict(state_dict_uri)
        os.chdir(self.wd_root_dict['current'])
        return state_dict

    def log_torch_state_dict(self, state_dict, save_dir):
        os.chdir(self.wd_root_dict['output'])
        with mlflow.start_run(run_id=self.run_id):
            pytorch.log_state_dict(state_dict, artifact_path=save_dir)
        os.chdir(self.wd_root_dict['current'])

    # other artifacts (To artifacts directory)
    def log_artifacts(self, target_dir, save_dir=None):
        os.chdir(self.wd_root_dict['output'])
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_artifacts(target_dir, artifact_path=save_dir)
        os.chdir(self.wd_root_dict['current'])

    def log_artifact(self, save_path):
        os.chdir(self.wd_root_dict['output'])
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_artifact(save_path)
        os.chdir(self.wd_root_dict['current'])

    # metric (To metrics directory)
    def log_metrics(self, metrics_dict, step=None):
        os.chdir(self.wd_root_dict['output'])
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_metrics(metrics_dict, step=step)
        os.chdir(self.wd_root_dict['current'])

    # params (To params directory)
    def log_params(self, params_dict):
        os.chdir(self.wd_root_dict['output'])
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_params(params_dict)
        os.chdir(self.wd_root_dict['current'])

    def log_params_from_omegaconf_dict(self, params, save_dir=None):
        params_dict = {}
        for param_name, element in params.items():
            tmp_params_dict = self.__explore_recursive(param_name, element)
            params_dict.update(tmp_params_dict)
        self.log_params(params_dict)
        self.log_artifacts(self.wd_root_dict['hydra'], save_dir)

    # tag (To tag directory)
    def log_tag(self, key, value):
        self.client.set_tag(self.run_id, key=key, value=value)

    def __explore_recursive(self, parent_name, element):
        tmp_params_dict = {}
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self.__explore_recursive(f'{parent_name}.{k}', v)
                else:
                    try:
                        tmp_params_dict[f'{parent_name}.{k}'] = v
                    except:
                        pass
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                tmp_params_dict[f'{parent_name}.{i}'] = v
        return tmp_params_dict

    def get_tracking_root(self):
        os.chdir(self.wd_root_dict['output'])
        with mlflow.start_run(run_id=self.run_id):
            tracking_root = mlflow.get_tracking_uri()
        os.chdir(self.wd_root_dict['current'])
        return tracking_root

    def get_artifact_root(self):
        os.chdir(self.wd_root_dict['output'])
        with mlflow.start_run(run_id=self.run_id):
            artifact_root = mlflow.get_artifact_uri()
        os.chdir(self.wd_root_dict['current'])
        return artifact_root

    def set_terminated(self, status):
        self.client.set_terminated(self.run_id, status=status)
