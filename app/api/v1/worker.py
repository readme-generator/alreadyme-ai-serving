from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time

import torch
from loguru import logger


class GenerationWorker:
    def __init__(
        self,
        model: str,
        max_length: int,
        top_k: int,
        temperature: float,
        no_repeat_ngram_size: int,
    ):
        self.model = model
        self.max_length = max_length
        self.top_k = top_k
        self.temperature = temperature
        self.no_repeat_ngram_size = no_repeat_ngram_size
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
            output = generator(
                prompt,
                do_sample=True,
                top_k=self.top_k,
                temperature=self.temperature,
                max_length=self.max_length,
            )
            output = output[0]["generated_text"][len(prompt) :]

            latency = time.time() - start_time
            logger.debug(f"generation complete. time: {latency}, length: {len(output)}")

            sender.send(output)
            sender.close()
            torch.cuda.empty_cache()
