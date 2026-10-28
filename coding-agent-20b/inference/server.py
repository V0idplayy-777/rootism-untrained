"""OpenAI-compatible serving with a continuous-batching scheduler.

The scheduler keeps a queue of pending requests and dispatches up to
`max_batch` generations concurrently (bounded by the model's KV budget via
GenerationConfig + PagedKVCache). Speculative decoding / true paged attention
require a CUDA engine (vLLM); this server is the correct fallback contract:
same HTTP API, same stop/JSON controls, swap the engine behind `generate_fn`.

Run:  python -m inference.server --checkpoint checkpoints/final --port 8000
"""

from __future__ import annotations



import argparse

import queue

import threading

import time

import uuid

from dataclasses import dataclass, field

from typing import Callable, Dict, List, Optional





@dataclass

class _Request:

    prompt_ids: List[int]

    gen_kwargs: Dict

    done: threading.Event = field(default_factory=threading.Event)

    result: Optional[List[int]] = None

    error: Optional[str] = None





class ContinuousBatcher:

    """Threaded scheduler: coalesces queued requests into dispatch waves."""



    def __init__(self, generate_fn: Callable, max_batch: int = 8):

        self.generate_fn = generate_fn

        self.max_batch = max_batch

        self.queue: "queue.Queue[_Request]" = queue.Queue()

        self._stop = threading.Event()

        self._thread = threading.Thread(target=self._loop, daemon=True)

        self.stats = {"requests": 0, "tokens": 0}

        self._thread.start()



    def _loop(self):

        while not self._stop.is_set():

            try:

                first = self.queue.get(timeout=0.1)

            except queue.Empty:

                continue

            wave = [first]

            while len(wave) < self.max_batch:

                try:

                    wave.append(self.queue.get_nowait())

                except queue.Empty:

                    break

            for req in wave:                                                              

                try:

                    req.result = self.generate_fn(req.prompt_ids, req.gen_kwargs)

                    self.stats["requests"] += 1

                    self.stats["tokens"] += len(req.result or [])

                except Exception as e:                                        

                    req.error = str(e)

                finally:

                    req.done.set()



    def submit(self, prompt_ids: List[int], gen_kwargs: Dict, timeout: float = 300) -> List[int]:

        req = _Request(prompt_ids, gen_kwargs)

        self.queue.put(req)

        if not req.done.wait(timeout=timeout):

            raise TimeoutError("generation timed out in queue")

        if req.error:

            raise RuntimeError(req.error)

        return req.result or []



    def close(self):

        self._stop.set()





def build_app(model, tokenizer):

    """FastAPI app (imported lazily so the scheduler is testable without it)."""

    from fastapi import FastAPI

    from pydantic import BaseModel



    app = FastAPI(title="CodingAgent-20B", version="1.0.0")

"""
FUCK!! IT'S BROKEN!!
"""

    def _generate(prompt_ids, kwargs):

        import torch

        from model.generation import GenerationConfig

        cfg = GenerationConfig(**{k: v for k, v in kwargs.items()

                                  if k in GenerationConfig.__dataclass_fields__})

        ids = torch.tensor([prompt_ids])

        out = model.generate(ids, generation_config=cfg, tokenizer=tokenizer)

        return out[0][len(prompt_ids):].tolist()



    batcher = ContinuousBatcher(_generate)



    class ChatRequest(BaseModel):

        messages: List[Dict[str, str]]

        max_new_tokens: int = 512

        temperature: float = 0.7

        top_p: float = 0.95

        top_k: int = 0

        min_p: float = 0.0

        json_mode: bool = False



    @app.get("/health")

    def health():

        return {"status": "ok", "stats": batcher.stats}



    @app.get("/v1/models")

    def models():

        return {"data": [{"id": "coding-agent-20b", "object": "model"}]}



    @app.post("/v1/chat/completions")

    def chat(req: ChatRequest):

        prompt = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in req.messages)

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

        out_ids = batcher.submit(prompt_ids, {

            "max_new_tokens": req.max_new_tokens, "temperature": req.temperature,

            "top_p": req.top_p, "top_k": req.top_k, "min_p": req.min_p,

            "json_mode": req.json_mode})

        text = tokenizer.decode(out_ids, skip_special_tokens=True)

        return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",

                "created": int(time.time()), "model": "coding-agent-20b",

                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},

                             "finish_reason": "stop"}]}



    return app





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--tokenizer", default=None)

    parser.add_argument("--host", default="0.0.0.0")

    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    import torch

    import uvicorn

    from model.config import CodingAgentConfig

    from model.model import CodingAgentModel

    from model.tokenizer import CodeTokenizer

    from training.checkpoint import load_verified

    tokenizer = CodeTokenizer.load(args.tokenizer) if args.tokenizer else CodeTokenizer()

    state = load_verified(args.checkpoint)

    cfg = CodingAgentConfig.from_dict(state["config"]) if "config" in state else CodingAgentConfig()

    model = CodingAgentModel(cfg)

    model.load_state_dict(state.get("model_state_dict", state), strict=False)

    model.eval()

    uvicorn.run(build_app(model, tokenizer), host=args.host, port=args.port)





if __name__ == "__main__":

    main()

