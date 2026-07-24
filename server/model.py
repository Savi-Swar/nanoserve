"""Thin wrapper over a HF causal LM exposing the two primitives every serving
engine is built from: prefill and single-step decode.

The batched engine generalizes exactly these two calls, so they're kept
explicit here.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .request import SamplingParams


def pick_device(pref: str | None = None) -> str:
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    # MPS fp16 matmul is buggy for these head dims (Metal "incompatible
    # dimensions"), so dev on Mac uses fp32. Real GPU runs use fp16 on CUDA.
    if device == "cuda":
        return torch.float16
    return torch.float32


class ModelRunner:
    def __init__(self, model_name: str, device: str | None = None, dtype=None):
        self.device = pick_device(device)
        self.dtype = dtype or pick_dtype(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        self.eos_id = self.tokenizer.eos_token_id
        self.model_name = model_name

    def encode(self, prompt: str) -> list[int]:
        return self.tokenizer(prompt, return_tensors=None)["input_ids"]

    def decode_text(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    # --- core primitives ------------------------------------------------
    @torch.no_grad()
    def prefill(self, input_ids: list[int]):
        """Run the full prompt. Returns (last_logits, kv_cache, cur_len)."""
        ids = torch.tensor([input_ids], device=self.device)
        n = ids.shape[1]
        out = self.model(
            input_ids=ids,
            past_key_values=DynamicCache(),
            use_cache=True,
            cache_position=torch.arange(n, device=self.device),
        )
        return out.logits[:, -1, :], out.past_key_values, n

    @torch.no_grad()
    def decode(self, token: int, cache, cur_len: int):
        """Advance one token. Returns (last_logits, kv_cache, new_len)."""
        ids = torch.tensor([[token]], device=self.device)
        out = self.model(
            input_ids=ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([cur_len], device=self.device),
        )
        return out.logits[:, -1, :], out.past_key_values, cur_len + 1

    @torch.no_grad()
    def decode_many(self, tokens: list[int], cache, cur: int):
        """Forward several tokens at once against the cache: the verification
        step of speculative decoding. Returns (logits[len,vocab], cache, new_len)
        where logits[i] predicts the token that follows tokens[i]."""
        n = len(tokens)
        ids = torch.tensor([tokens], device=self.device)
        out = self.model(
            input_ids=ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(cur, cur + n, device=self.device),
        )
        return out.logits[0], out.past_key_values, cur + n

    def crop_cache(self, cache, length: int):
        """Truncate the KV cache to `length` tokens (drop rejected speculation)."""
        for l in cache.layers:
            l.keys = l.keys[:, :, :length, :]
            l.values = l.values[:, :, :length, :]
        return cache

    def sync(self):
        """Block until queued device work is done; required before timing."""
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def warmup(self, n: int = 3):
        """Compile kernels / page in weights so the first real request isn't
        an outlier. MPS especially JITs on first use."""
        ids = self.encode("Warmup.")
        for _ in range(n):
            logits, kv, cur = self.prefill(ids)
            tok = int(logits.argmax(-1))
            for _ in range(4):
                logits, kv, cur = self.decode(tok, kv, cur)
                tok = int(logits.argmax(-1))
        self.sync()


@torch.no_grad()
def sample(logits: torch.Tensor, params: SamplingParams) -> int:
    """logits: [1, vocab]. Returns a single token id."""
    if params.greedy:
        return int(logits.argmax(-1))
    logits = logits / params.temperature
    probs = torch.softmax(logits, dim=-1)
    if params.top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        # keep the smallest prefix whose cumulative mass exceeds top_p
        drop = (cum - sp) > params.top_p
        sp = sp.masked_fill(drop, 0.0)
        sp = sp / sp.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sp, 1)
        return int(si.gather(-1, choice))
    return int(torch.multinomial(probs, 1))
