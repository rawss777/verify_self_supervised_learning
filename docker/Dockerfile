# Ubuntu + CUDA11.3環境を使う
# https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

# Ubuntuパッケージのインストール
RUN apt-get update && apt-get install -y \
    wget

# Minicondaのインストール
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    /opt/miniconda/bin/conda create -n py39 python=3.9 && \
    /opt/miniconda/bin/conda clean --all -y && \
    rm -r Miniconda3-latest-Linux-x86_64.sh

# 用意したconda環境をデフォルト環境に設定
ENV PATH /opt/miniconda/envs/py39/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/miniconda/envs/py39/lib:$LD_LIBRARY_PATH

# pipパッケージのインストール
COPY requirements.txt /requirements.txt
RUN chmod +x /requirements.txt && \
    pip install -r requirements.txt && \
    rm /requirements.txt
