"""Turn a list of finished Requests into the numbers that tell the story:
throughput, TTFT tail, end-to-end latency tail, queue delay, GPU util.
"""
from __future__ import annotations

from dataclasses import dataclass

from server.request import Request


def pct(xs: list[float], p: float) -> float:
    """Linear-interpolated percentile. p in [0,100]."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class Report:
    engine: str
    device: str
    n: int
    wall: float
    out_tokens: int
    throughput: float          # output tok/s across the whole run
    ttft: dict                 # p50/p90/p99 seconds
    e2e: dict
    queue: dict
    per_req_decode_tps: float  # median steady-state decode speed
    gpu: dict | None

    def to_dict(self, model: str = "", rate: float = 0.0, max_tokens: int = 0) -> dict:
        return {
            "engine": self.engine,
            "device": self.device,
            "model": model,
            "rate": rate,
            "n": self.n,
            "max_tokens": max_tokens,
            "wall": self.wall,
            "out_tokens": self.out_tokens,
            "throughput": self.throughput,
            "per_req_decode_tps": self.per_req_decode_tps,
            "ttft": self.ttft,
            "e2e": self.e2e,
            "queue": self.queue,
            "gpu": self.gpu,
        }

    def render(self) -> str:
        def ms(d):
            return f"p50={d['p50']*1e3:6.0f}ms  p90={d['p90']*1e3:6.0f}ms  p99={d['p99']*1e3:6.0f}ms"

        lines = [
            f"engine={self.engine}  device={self.device}  requests={self.n}",
            f"wall={self.wall:.2f}s  out_tokens={self.out_tokens}",
            f"THROUGHPUT       {self.throughput:8.1f} tok/s",
            f"decode (median)  {self.per_req_decode_tps:8.1f} tok/s/req",
            f"TTFT             {ms(self.ttft)}",
            f"end-to-end       {ms(self.e2e)}",
            f"queue delay      {ms(self.queue)}",
        ]
        if self.gpu:
            lines.append(
                f"GPU util         mean={self.gpu['mean']:.0f}%  peak={self.gpu['peak']:.0f}%"
            )
        else:
            lines.append("GPU util         n/a (no NVIDIA GPU)")
        return "\n".join(lines)


def build_report(
    engine_name: str, device: str, reqs: list[Request], wall: float, gpu=None
) -> Report:
    done = [r for r in reqs if r.finish_time is not None]
    out_tokens = sum(r.num_output for r in done)
    ttft = [r.ttft for r in done]
    e2e = [r.e2e for r in done]
    queue = [r.queue_delay for r in done]
    decode = sorted(r.decode_tps for r in done if r.decode_tps > 0)
    med_decode = decode[len(decode) // 2] if decode else 0.0

    trio = lambda xs: {"p50": pct(xs, 50), "p90": pct(xs, 90), "p99": pct(xs, 99)}
    return Report(
        engine=engine_name,
        device=device,
        n=len(done),
        wall=wall,
        out_tokens=out_tokens,
        throughput=out_tokens / wall if wall > 0 else 0.0,
        ttft=trio(ttft),
        e2e=trio(e2e),
        queue=trio(queue),
        per_req_decode_tps=med_decode,
        gpu=gpu,
    )
