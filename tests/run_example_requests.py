from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from typing import Any

from aiohttp import ClientSession, WSMsgType

URL_GENERATION_REQUEST = "/api/v2/generate/request"
URL_GENERATION_STREAM = "/api/v2/generate/stream"


async def stream_and_save(session: ClientSession, base_url: str, request_id: str):
    with open(f"{request_id}.md", "w") as fp:
        async with session.ws_connect(base_url + URL_GENERATION_STREAM) as ws:
            await ws.send_str(request_id)
            async for msg in ws:
                if (
                    msg.type in (WSMsgType.CLOSED, WSMsgType.ERROR)
                    or msg.type == WSMsgType.TEXT
                    and msg.data == ""
                ):
                    break
                fp.write(msg.data)


async def request_and_start_tasks(
    args: argparse.Namespace, example_input: dict[str, Any]
):
    async with ClientSession() as session:
        tasks = []
        for _ in range(args.max_stream):
            request_id = generate_request_id()
            await session.post(
                args.base_url + URL_GENERATION_REQUEST,
                json={"requestId": request_id, **example_input},
            )
            coro = stream_and_save(session, args.base_url, request_id)
            tasks.append(asyncio.create_task(coro))
        await asyncio.wait(tasks)


def generate_request_id() -> str:
    return "".join(random.choices("0123456789abcdef", k=8))


def main(args: argparse.Namespace):
    example_input = os.path.join(os.path.dirname(__file__), "example_input.json")
    with open(args.example_input or example_input) as fp:
        example_input = json.load(fp)
    asyncio.run(request_and_start_tasks(args, example_input))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://0.0.0.0:8080")
    parser.add_argument("--example-input")
    parser.add_argument("--max-stream", type=int, default=8)
    main(parser.parse_args())
