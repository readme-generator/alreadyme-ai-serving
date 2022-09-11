ARG CUDA_VER=11.6.1
ARG CUDNN_VER=8
ARG UBUNTU_VER=18.04
FROM nvidia/cuda:${CUDA_VER}-cudnn${CUDNN_VER}-devel-ubuntu${UBUNTU_VER}

LABEL maintainer="ALREADYME"
LABEL repository="alreadyme-ai-serving"
LABEL version="v0.1.0"

RUN apt update && \
    apt install -y wget \
                   build-essential \
                   ca-certificates && \
    rm -rf /var/lib/apt/lists

ARG PYTHON_VER=39
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py${PYTHON_VER}_4.12.0-Linux-x86_64.sh && \
    bash Miniconda3-py${PYTHON_VER}_4.12.0-Linux-x86_64.sh -p /miniconda -b && \
    rm Miniconda3-py${PYTHON_VER}_4.12.0-Linux-x86_64.sh

ENV PATH=/miniconda/bin:${PATH}
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu113 \
        torch \
        "fastapi[all]" \
        omegaconf \
        transformers \
        loguru

COPY ./app /workspace/app
WORKDIR /workspace/app

EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]