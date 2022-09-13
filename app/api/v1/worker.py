from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time
from typing import Any

import torch
from loguru import logger


class GenerationWorker:
    def __init__(self, model: str, **kwargs: Any):
        self.model = model
        self.generation_kwargs = kwargs
        self.queue = mp.Queue()

    def request(self, prompt: str) -> asyncio.Future:
        receiver, sender = mp.Pipe(duplex=False)
        self.queue.put_nowait((prompt, sender))

        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, receiver.recv)

    def run(self) -> mp.Process:
        process = mp.Process(target=self.worker_fn, daemon=True)
        process.start()
        return process

    def worker_fn(self):
        start_time = time.time()
        # generator = pipeline(
        #     task="text-generation",
        #     model=self.model,
        #     low_cpu_mem_usage=True,
        #     torch_dtype=torch.float16,
        #     device=0,
        # )
        generator = torch.load(os.path.join(self.model, "pipeline.pt"))

        latency = time.time() - start_time
        logger.debug(f"generator [{self.model}] has been initialized. time: {latency}")

        while True:
            prompt, sender = self.queue.get()
            logger.debug("generation request is received in worker process.")

            start_time = time.time()
            output = generator(prompt, **self.generation_kwargs)
            output = output[0]["generated_text"][len(prompt) :]

            latency = time.time() - start_time
            logger.debug(f"generation complete. time: {latency}, length: {len(output)}")

            sender.send(output)
            sender.close()
            torch.cuda.empty_cache()
