from __future__ import annotations

import os

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from .utils import create_input_prompt
from .worker import GenerationWorker

os.environ["TOKENIZERS_PARALLELISM"] = "false"

router = APIRouter()
config = OmegaConf.load(os.environ.get("SERVER_CONFIG", "resources/config.yaml"))
tokenizer = AutoTokenizer.from_pretrained(config.model)

worker = GenerationWorker(config.model, **config.generation)
worker.run()


class RepositoryData(BaseModel):
    githubOriginalUrl: str = Field(
        example="https://github.com/readme-generator/alreadyme-ai-serving.git"
    )
    data: dict[str, str] = Field(
        example={
            "main.py": "import os\nimport glob\n...",
            "utils.py": "from typing import Any\n...",
        }
    )


@router.post(
    "/generate",
    summary="Generate README content from repository.",
    response_class=PlainTextResponse,
)
async def generate(repository: RepositoryData) -> str:
    """
    This API is for generating `README.md` content from repository files. This version
    of the model needs git-URL (e.g.
    https://github.com/readme-generator/alreadyme-ai-serving.git) to give information
    about URLs and the repository name. As mentioned in the parameter section, you have
    to send `githubOriginalUrl` parameter. We strongly recommend to use the URL which
    ends with `.git` because our **dataset version 3** uses the full git-URLs.
    """
    prompt = create_input_prompt(
        repository=repository.githubOriginalUrl,
        files=repository.data,
        tokenizer=tokenizer,
        separator=config.prompt.separator,
        max_total_tokens=config.prompt.max_total_tokens,
        max_source_code_tokens=config.prompt.max_source_code_tokens,
    )
    output = await worker.request(prompt)
    output = output.split(config.prompt.separator)[0]
    return output
