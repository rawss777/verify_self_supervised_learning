#!/bin/bash

function fine_tune_and_test(){
  python fine_tune.py \
  experiment_params.pretrain_run_name=$1 experiment_params.run_name=$1"-fix_encoder" \
  fine_tune_params.model_state="last" fine_tune_params.epoch_num=$2 \
  fine_tune_params.fix_encoder=true

  python test.py \
  experiment_params.fine_tune_run_name=$1"-fix_encoder" experiment_params.run_name=$1"-fix_encoder" \
  test_params.model_state="best"

  python fine_tune.py \
  experiment_params.pretrain_run_name=$1 experiment_params.run_name=$1"-fine_tune" \
  fine_tune_params.model_state="last" fine_tune_params.epoch_num=$2 \
  fine_tune_params.fix_encoder=false

  python test.py \
  experiment_params.fine_tune_run_name=$1"-fine_tune" experiment_params.run_name=$1"-fine_tune" \
  test_params.model_state="best"
}

# # # Common
seed=0
dataset="cifar10"
output_dir="output/"$dataset
ssl_epoch=300
fine_tune_epoch=100
batch_size=512


# # # BYOL
ssl_name="byol"
loss_fn="cosine_similarity_loss"
run_name="test_"$ssl_name
# train
python train.py \
self_supervised=$ssl_name augmentation=$ssl_name loss_fn=$loss_fn \
experiment_params.run_name=$run_name train_params.epoch_num=$ssl_epoch train_params.batch_size=$batch_size \
self_supervised.params.use_eman=true
# test
fine_tune_and_test $run_name $fine_tune_epoch


# # # MoCo
ssl_name="moco"
loss_fn="info_nce_loss"
run_name="test_"$ssl_name
# train
python train.py \
self_supervised=$ssl_name augmentation=$ssl_name loss_fn=$loss_fn \
experiment_params.run_name=$run_name train_params.epoch_num=$ssl_epoch train_params.batch_size=$batch_size \
self_supervised.params.use_eman=true
# test
fine_tune_and_test $run_name $fine_tune_epoch


# # # SimCLR
ssl_name="simclr"
loss_fn="info_nce_loss"
# train
run_name="test_"$ssl_name
python train.py \
self_supervised=$ssl_name augmentation=$ssl_name loss_fn=$loss_fn \
experiment_params.run_name=$run_name train_params.epoch_num=$ssl_epoch train_params.batch_size=$batch_size
# test
fine_tune_and_test $run_name $fine_tune_epoch


# # # SwAV
ssl_name="swav"
loss_fn="soft_cross_entropy_loss"
# train
run_name="test_"$ssl_name
python train.py \
self_supervised=$ssl_name augmentation=$ssl_name loss_fn=$loss_fn \
experiment_params.run_name=$run_name train_params.epoch_num=$ssl_epoch train_params.batch_size=$batch_size
# test
fine_tune_and_test $run_name $fine_tune_epoch


# # # SimSIAM
ssl_name="simsiam"
loss_fn="cosine_similarity_loss"
# train
run_name="test_"$ssl_name
python train.py \
self_supervised=$ssl_name augmentation=$ssl_name loss_fn=$loss_fn \
experiment_params.run_name=$run_name train_params.epoch_num=$ssl_epoch train_params.batch_size=$batch_size
# test
fine_tune_and_test $run_name $fine_tune_epoch


# # # DeepClusterV2
ssl_name="deepclusterv2"
loss_fn="with_temp_cross_entropy_loss"
# train
run_name="test_"$ssl_name
python train.py \
self_supervised=$ssl_name augmentation=$ssl_name loss_fn=$loss_fn \
experiment_params.run_name=$run_name train_params.epoch_num=$ssl_epoch train_params.batch_size=$batch_size
# test
fine_tune_and_test $run_name $fine_tune_epoch

