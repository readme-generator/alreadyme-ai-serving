# Serving ALREADYME.md AI Model

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?&logo=fastapi)](https://github.com/tiangolo/fastapi)
[![docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![redoc](https://img.shields.io/badge/docs-redoc-blue)](https://jolly-zebra-41.redoc.ly/)
[![license](https://img.shields.io/github/license/readme-generator/alreadyme-ai-research)](./LICENSE)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CodeFactor](https://www.codefactor.io/repository/github/readme-generator/alreadyme-ai-serving/badge)](https://www.codefactor.io/repository/github/readme-generator/alreadyme-ai-serving)

This repository is to serve [ALREADYME model](https://github.com/readme-generator/alreadyme-ai-research) on FastAPI.

## Requirements
* torch
* fastapi[all]
* omegaconf
* transformers
* loguru

## Prerequisites
Before starting the server, the fine-tuned model weight is required. While `transformers` pipeline has extremely slow, we use pickling to enhance the initialization time. Because of that, some conversion is needed:
```python
import torch
from transformers import pipeline

pipe = pipeline("text-generation", "bloom-1b7-finetuned-readme-270k-steps", torch_dtype=torch.float16, device=0)
torch.save(pipe, "bloom-1b7-finetuned-readme-270k-steps/pipeline.pt")
```

Move the transformer model to `app/resources` and change the path in `app/resources/config.yaml`.

## Run the server

We recommend to build a docker image instead using in local. But it would be better to run before building the image to check any bug in the code and your fine-tuned model.

### Start locally
```bash
$ cd app
$ uvicorn main:app --ip [your ip address] --port [your port]
```

### Build docker
We do not provide any pre-build image yet. Build your own image with custom fine-tuned model!

```bash
$ docker build -t alreadyme-ai-serving:v0.1.2 -f Dockerfile \
    --build-args CUDA_VER=11.6.1 \
    --build-args CUDNN_VER=8 \
    --build-args UBUNTU_VER=18.04 \
    --build-args PYTHON_VER=39
```

You can change the version of cuda, cudnn, ubuntu and python. They can be useful for compatibility of different cloud environment. After build your image, run docker by:

```bash
$ docker run --gpus all -p 8080:80 alreadyme-ai-serving:v0.1.2
```
The docker container will launch the server on port 80, so you should binding to your own port number (e.g. 8080).

## Documentation
**alreadyme-ai-serving** supports OpenAPI and you can see the documentation of the APIs in your server. If the server is running locally, check out `http://127.0.0.1:8080/docs` for swagger or `http://127.0.0.1:8080/redoc` for redoc.

For convenience, we hosted [free redoc documentation page](https://jolly-zebra-41.redoc.ly/). You may login to see the details.

## License
**alreadyme-ai-serving** is released under the Apache License 2.0. License can be found in [here](./LICENSE).
