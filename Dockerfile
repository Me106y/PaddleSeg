# PaddleSeg 训练镜像
#
# 目标：把「本仓库 + 自定义模型/配置 + PP-HumanSeg14K 训练管线」打包成一个可直接在
# GPU 服务器上跑训练的镜像，省掉在服务器上重配环境。
#
# 构建：
#     docker build -t paddleseg-train:latest .
#
# 运行（data / output / log 用挂载，不进镜像）：
#     docker run --gpus all -it --rm --shm-size=8g \
#       -v "$PWD/data:/workspace/data" \
#       -v "$PWD/output:/workspace/output" \
#       -v "$PWD/log:/workspace/log" \
#       paddleseg-train:latest \
#       python tools/train.py \
#         --config configs/quick_start/repvit_v8_20.yml \
#         --save_dir output/repvit_v8_20 \
#         --do_eval --use_vdl --save_interval 500
#
# 前提：宿主机已装 NVIDIA 驱动 + nvidia-container-toolkit（`--gpus all` 才可用）。
#
# 关于基础镜像标签：paddlepaddle/paddle 的标签形如
#     <paddle版本>-gpu-cuda<CUDA版本>-cudnn<cudnn版本>[-trt<TRT版本>]
# 下面的默认值对应本地环境（paddle 3.0.0）。若拉取失败，先按上面形确认一个可用标签，
# 再用 --build-arg 覆盖，例如：
#     docker build --build-arg BASE_IMAGE=paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5 .

ARG BASE_IMAGE=paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/workspace \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLAGS_allocator_strategy=auto_growth

# OpenCV / 可视化依赖的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 先装依赖：requirements.txt 没变时这一层可复用，改代码不会触发重装
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# 再拷代码（.dockerignore 已排除 data/ output/ log/ .git 等）
COPY . /workspace

# 冒烟自检：确认 import 到的是镜像里的 paddleseg，而不是 site-packages 里的上游副本。
# 本仓库的 tools/*.py 依赖「仓库根在 sys.path 上」，否则会报 XXX was not found。
RUN python -c "import os, paddleseg; p = os.path.dirname(paddleseg.__file__); \
    assert p.startswith('/workspace'), 'imported foreign paddleseg: ' + p; \
    print('paddleseg OK ->', p)" \
    && python -c "import paddle; print('paddle', paddle.__version__, '| cuda', paddle.is_compiled_with_cuda())"

CMD ["python", "tools/train.py", \
     "--config", "configs/quick_start/repvit_v8_20.yml", \
     "--save_dir", "output/repvit_v8_20", \
     "--do_eval"]
