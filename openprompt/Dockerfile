# 阶段一: 选择一个包含 CUDA 12.1 和 cuDNN 的官方基础镜像
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 阶段二: 设置基本环境和系统依赖
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    sed \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 创建 python3 -> python 软链接
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# 阶段三: 创建 Python 虚拟环境
ENV VENV_PATH=/opt/venv
RUN python3.11 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# 阶段四: 安装所有 Python 依赖库
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
RUN pip install torch_geometric==2.4.0 torch_sparse torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
RUN pip install -r requirements.txt

# 阶段五: 设置工作目录并定义默认启动命令
WORKDIR /workspace
# (核心修正) 移除 ENTRYPOINT，设置一个完整的默认 CMD
CMD ["python", "fewshot-config.py"]