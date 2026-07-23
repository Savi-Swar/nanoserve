"""HTTP inference service around the continuous-batching engine.

    POST /generate  {"prompt": "...", "max_tokens": 64}  -> {"text", "tokens", "ttft_ms"}
    GET  /metrics   Prometheus-style text
    GET  /healthz   200 ok

Concurrent HTTP clients are submitted into one shared engine queue, so the
continuous batcher serves them together. The server admits a request only if the
queue depth is under a limit; past it, it sheds load with a 503 instead of
letting the queue grow without bound and blowing every request's latency. That
backpressure is the difference between a server and a benchmark loop.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler

from .engine import ENGINES
from .request import Request, SamplingParams


class _Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.accepted = 0
        self.completed = 0
        self.shed = 0
        self.out_tokens = 0
        self.ttft = deque(maxlen=256)   # recent TTFT (seconds)
        self.start = time.perf_counter()


class InferenceService:
    def __init__(self, model, engine="continuous", max_batch=16, max_queue=48,
                 num_blocks=8192, block_size=16):
        self.m = model
        self.engine_name = engine
        self.max_queue = max_queue
        self.metrics = _Metrics()
        self._events: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self._ctr = 0

        cfg = {"max_batch": max_batch}
        if engine in ("paged", "spec_cont"):
            cfg["num_blocks"] = num_blocks
        self.engine = ENGINES[engine](model, on_finish=self._on_finish, **cfg)
        self.engine.start()

    def queue_depth(self) -> int:
        active = self.engine.state.size if hasattr(self.engine, "state") else 0
        return self.engine._q.qsize() + active

    def _on_finish(self, req: Request):
        with self.metrics.lock:
            self.metrics.completed += 1
            self.metrics.out_tokens += req.num_output
            if req.first_token_time and req.arrival_time:
                self.metrics.ttft.append(req.ttft)
        with self._lock:
            ev = self._events.get(req.id)
        if ev:
            ev.set()

    def generate(self, prompt: str, max_tokens: int = 64):
        """Blocking submit-and-wait. Returns None if the server is shedding load."""
        if self.queue_depth() >= self.max_queue:
            with self.metrics.lock:
                self.metrics.shed += 1
            return None
        with self._lock:
            rid = self._ctr
            self._ctr += 1
            ev = threading.Event()
            self._events[rid] = ev
        with self.metrics.lock:
            self.metrics.accepted += 1
        req = Request(rid, prompt,
                      SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=False))
        self.engine.submit(req)
        ev.wait(timeout=120)
        with self._lock:
            self._events.pop(rid, None)
        return {
            "text": self.m.decode_text(req.output_tokens),
            "tokens": req.num_output,
            "ttft_ms": round(req.ttft * 1000, 1) if req.first_token_time else None,
        }

    def prometheus(self) -> str:
        with self.metrics.lock:
            elapsed = time.perf_counter() - self.metrics.start
            tput = self.metrics.out_tokens / elapsed if elapsed > 0 else 0.0
            ttfts = sorted(self.metrics.ttft)
            p99 = ttfts[min(len(ttfts) - 1, int(len(ttfts) * 0.99))] * 1e3 if ttfts else 0.0
            acc, comp, shed, out = (self.metrics.accepted, self.metrics.completed,
                                    self.metrics.shed, self.metrics.out_tokens)
        rows = [
            ("nanoserve_requests_accepted_total", acc),
            ("nanoserve_requests_completed_total", comp),
            ("nanoserve_requests_shed_total", shed),
            ("nanoserve_queue_depth", self.queue_depth()),
            ("nanoserve_output_tokens_total", out),
            ("nanoserve_throughput_tokens_per_second", round(tput, 1)),
            ("nanoserve_ttft_p99_ms", round(p99, 1)),
        ]
        return "".join(f"{k} {v}\n" for k, v in rows)

    def stop(self):
        self.engine.stop()


def make_handler(service: InferenceService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):  # keep stdout to our own logs
            pass

        def _send(self, code, body, ctype="application/json"):
            b = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/healthz":
                self._send(200, "ok\n", "text/plain")
            elif self.path == "/metrics":
                self._send(200, service.prometheus(), "text/plain; version=0.0.4")
            else:
                self._send(404, "not found\n", "text/plain")

        def do_POST(self):
            if self.path != "/generate":
                self._send(404, '{"error":"not found"}')
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                self._send(400, '{"error":"bad json"}')
                return
            out = service.generate(str(body.get("prompt", "")),
                                   int(body.get("max_tokens", 64)))
            if out is None:
                self._send(503, '{"error":"overloaded; queue full, retry later"}')
            else:
                self._send(200, json.dumps(out))

    return Handler
