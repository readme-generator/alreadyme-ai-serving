from __future__ import annotations

import os

from aiohttp import ClientSession
from fastapi import APIRouter, WebSocket
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from .generation import GenerationOptions, GenerationSession
from .utils import create_input_prompt

router = APIRouter()

config: DictConfig
tokenizer: PreTrainedTokenizerBase
session: GenerationSession


class GenerationRequestData(BaseModel):
    requestId: str = Field(example="7def3b02")
    githubOriginalUrl: str = Field(
        example="https://github.com/readme-generator/alreadyme-ai-serving.git"
    )
    data: dict[str, str] = Field(
        example={
            "main.py": "import os\nimport glob\n...",
            "utils.py": "from typing import Any\n...",
        }
    )


@router.on_event("startup")
async def startup():
    global config, tokenizer, session

    config = os.environ.get("SERVER_CONFIG_V2", "resources/config-v2.yaml")
    config = OmegaConf.load(config)

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    model_config = AutoConfig.from_pretrained(config.model)
    model_path = os.path.join(config.model, "model.pt")

    options = GenerationOptions(
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.n_layer,
        num_attention_heads=model_config.n_head,
        max_stream=config.generation.max_stream,
        temperature=config.generation.temperature,
        pad_to_multiple_of=config.generation.pad_to_multiple_of,
        max_length=config.generation.max_length,
    )
    session = GenerationSession(model_path, tokenizer, options)
    session.initialize()


@router.on_event("shutdown")
async def shutdown():
    session.close()


@router.post(
    "/generate/request",
    summary="Request for README content generation from repository.",
)
async def request_generation(request: GenerationRequestData) -> str:
    """
    This API is for generating `README.md` content from repository files. This version
    of the model needs git-URL (e.g.
    https://github.com/readme-generator/alreadyme-ai-serving.git) to give information
    about URLs and the repository name. As mentioned in the parameter section, you have
    to send `githubOriginalUrl` parameter. We strongly recommend to use the URL which
    ends with `.git` because our **dataset version 3** uses the full git-URLs. After
    calling this API, you can receive the generated words through WebSocket stream.
    """
    input_prompt = create_input_prompt(
        repository=request.githubOriginalUrl,
        files=request.data,
        tokenizer=tokenizer,
        separator=config.prompt.separator,
        max_total_tokens=config.prompt.max_total_tokens,
        max_source_code_tokens=config.prompt.max_source_code_tokens,
    )
    session.request(request.requestId, input_prompt)


@router.websocket("/generate/stream")
async def stream_generation(websocket: WebSocket):
    """
    This API is an endpoint of websocket connection for receiving the generation results
    in real time. First, client should send its request id which it wants to subscribe.
    Then the words would be received through the websocket. Note that the empty string
    will be sended for closing the stream. That is, if the sentence generation is
    completed then the empty string will be arrived.
    """
    await websocket.accept()
    request_id = await websocket.receive_text()

    text = ""
    async for word in session.stream(request_id):
        text += word
        await websocket.send_text(word)
    await websocket.close()

    if config.callback is not None:
        async with ClientSession() as http:
            await http.put(config.callback, json={"id": request_id, "readmeText": text})
