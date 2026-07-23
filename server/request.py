"""Request + sampling types shared across every engine."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 0.0  # 0.0 => greedy (deterministic, best for benchmarking)
    top_p: float = 1.0
    ignore_eos: bool = False  # keep generating until max_tokens (stable token counts)

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


@dataclass
class Request:
    id: int
    prompt: str
    sampling: SamplingParams
    prompt_ids: list[int] | None = None  # pre-tokenized (trace replay uses exact lengths)

    def input_ids(self, model) -> list[int]:
        """Token ids for this request — pre-tokenized if supplied (trace
        replay), else encode the prompt text."""
        return self.prompt_ids if self.prompt_ids is not None else model.encode(self.prompt)

    # --- filled in as the request flows through the system ---
    arrival_time: float | None = None      # when the load generator released it
    schedule_time: float | None = None     # when an engine first touched it
    first_token_time: float | None = None  # TTFT anchor
    finish_time: float | None = None
    prompt_len: int = 0
    output_tokens: list[int] = field(default_factory=list)

    @property
    def num_output(self) -> int:
        return len(self.output_tokens)

    # latency breakdown (all seconds) -----------------------------------
    @property
    def ttft(self) -> float:
        return self.first_token_time - self.arrival_time

    @property
    def queue_delay(self) -> float:
        return self.schedule_time - self.arrival_time

    @property
    def e2e(self) -> float:
        return self.finish_time - self.arrival_time

    @property
    def decode_tps(self) -> float:
        """Steady-state tokens/sec for this request alone (excludes TTFT)."""
        gen = self.finish_time - self.first_token_time
        n = self.num_output - 1
        return n / gen if gen > 0 and n > 0 else 0.0

    @property
    def tpot(self) -> float:
        """Time per output token (seconds) — the inter-token latency a user
        feels once streaming starts. The p99 of this is a standard SLO."""
        gen = self.finish_time - self.first_token_time
        n = self.num_output - 1
        return gen / n if n > 0 else 0.0

    def meets_slo(self, ttft_s: float, tpot_s: float) -> bool:
        return (self.finish_time is not None
                and self.ttft <= ttft_s and self.tpot <= tpot_s)
