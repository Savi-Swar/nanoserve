"""HTTP inference service around the continuous-batching engine.

    POST /generate  {"prompt": "...", "max_tokens": 64}            -> {"text", ...}
    POST /generate  {"prompt": "...", "stream": true}              -> SSE token stream
    GET  /metrics   Prometheus-style text
    GET  /healthz   200 ok  (503 while draining)

Request lifecycle is the point here, not raw throughput:

- Backpressure / load shedding: a request is admitted only while the engine
  queue is under a limit; past it the server returns 503 instead of letting the
  queue grow without bound and wrecking every request's latency.
- Cancellation on disconnect: a streaming client that hangs up makes the next
  write fail; the handler cancels the request, and the engine evicts it from the
  running batch mid-step and returns its KV blocks to the free list.
- Streaming backpressure: each stream has a bounded token buffer; a client that
  reads slower than the model generates overflows it and gets shed, rather than
  back-pressuring the shared batch.
- Timeouts: an optional per-request deadline; the engine evicts on breach.
- Graceful shutdown: `drain()` stops admitting and lets in-flight work finish.
- Structured logging: every request's lifecycle (queued -> scheduled ->
  first_token -> finished/cancelled/timeout) is traceable by id.
"""
from __future__ import annotations

import json
import queue
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
        self.cancelled = 0
        self.timed_out = 0
        self.out_tokens = 0
        self.ttft = deque(maxlen=256)   # recent TTFT (seconds)
        self.start = time.perf_counter()


class _Stream:
    """A live SSE connection's buffer: bounded token queue + a done signal."""
    __slots__ = ("q", "done", "status", "ids", "sent")

    def __init__(self, maxbuf: int):
        self.q: "queue.Queue[str]" = queue.Queue(maxsize=maxbuf)
        self.done = threading.Event()
        self.status = "running"
        self.ids: list[int] = []
        self.sent = ""


class InferenceService:
    def __init__(self, model, engine="continuous", max_batch=16, max_queue=48,
                 num_blocks=8192, block_size=16, request_timeout=None,
                 stream_buffer=1024, log=False):
        self.m = model
        self.engine_name = engine
        self.max_queue = max_queue
        self.request_timeout = request_timeout   # seconds, or None
        self.stream_buffer = stream_buffer
        self.metrics = _Metrics()
        self._events: dict[int, threading.Event] = {}
        self._streams: dict[int, _Stream] = {}
        self._lock = threading.Lock()
        self._ctr = 0
        self.draining = False
        self._log = log

        cfg = {"max_batch": max_batch}
        if engine in ("paged", "spec_cont"):
            cfg["num_blocks"] = num_blocks
        self.engine = ENGINES[engine](
            model, on_finish=self._on_finish, on_token=self._on_token,
            on_event=self._on_event if log else None, **cfg)
        self.engine.start()

    # --- capacity ------------------------------------------------------
    def queue_depth(self) -> int:
        active = self.engine.state.size if hasattr(self.engine, "state") else 0
        return self.engine._q.qsize() + active

    def _next_id(self) -> int:
        with self._lock:
            rid = self._ctr
            self._ctr += 1
        return rid

    def _admit(self, prompt, max_tokens):
        """Shared admission: shed if draining or overloaded, else build+submit a
        request. Returns the Request or None (shed)."""
        if self.draining or self.queue_depth() >= self.max_queue:
            with self.metrics.lock:
                self.metrics.shed += 1
            return None
        rid = self._next_id()
        req = Request(rid, str(prompt),
                      SamplingParams(max_tokens=int(max_tokens), temperature=0.0,
                                     ignore_eos=False))
        if self.request_timeout is not None:
            req.deadline = time.perf_counter() + self.request_timeout
        with self.metrics.lock:
            self.metrics.accepted += 1
        return req

    # --- engine callbacks ---------------------------------------------
    def _on_token(self, req: Request, tok: int):
        st = self._streams.get(req.id)
        if st is None:
            return
        st.ids.append(tok)
        full = self.m.decode_text(st.ids)
        delta, st.sent = full[len(st.sent):], full
        if not delta:
            return
        try:
            st.q.put_nowait(delta)
        except queue.Full:
            # client can't keep up -> shed it rather than stall the batch
            self.engine.cancel(req.id)

    def _on_finish(self, req: Request):
        with self.metrics.lock:
            self.metrics.completed += 1
            self.metrics.out_tokens += req.num_output
            if req.status == "cancelled":
                self.metrics.cancelled += 1
            elif req.status == "timeout":
                self.metrics.timed_out += 1
            if req.first_token_time and req.arrival_time:
                self.metrics.ttft.append(req.ttft)
        st = self._streams.get(req.id)
        if st is not None:
            st.status = req.status
            st.done.set()
        with self._lock:
            ev = self._events.get(req.id)
        if ev:
            ev.set()

    def _on_event(self, name: str, req: Request):
        t = time.perf_counter() - self.metrics.start
        extra = ""
        if name in ("finish", "cancelled", "timeout"):
            ttft = f"{req.ttft * 1e3:.0f}ms" if req.first_token_time else "-"
            extra = f" tokens={req.num_output} ttft={ttft}"
        elif name == "scheduled":
            extra = f" qdepth={self.queue_depth()}"
        print(f"[t=+{t:7.3f}s] rid={req.id} {name}{extra}", flush=True)

    # --- non-streaming -------------------------------------------------
    def generate(self, prompt: str, max_tokens: int = 64):
        req = self._admit(prompt, max_tokens)
        if req is None:
            return None
        ev = threading.Event()
        with self._lock:
            self._events[req.id] = ev
        self.engine.submit(req)
        ev.wait(timeout=300)
        with self._lock:
            self._events.pop(req.id, None)
        return {
            "text": self.m.decode_text(req.output_tokens),
            "tokens": req.num_output,
            "status": req.status,
            "ttft_ms": round(req.ttft * 1000, 1) if req.first_token_time else None,
        }

    # --- streaming -----------------------------------------------------
    def begin_stream(self, prompt: str, max_tokens: int = 64):
        req = self._admit(prompt, max_tokens)
        if req is None:
            return None, None
        st = _Stream(self.stream_buffer)
        with self._lock:
            self._streams[req.id] = st
        self.engine.submit(req)
        return req.id, st

    def stream_tokens(self, rid: int, st: _Stream):
        """Yield decoded token pieces until the request finishes. The caller
        writes them to the client and cancels `rid` if the write fails."""
        try:
            while True:
                try:
                    yield st.q.get(timeout=0.25)
                except queue.Empty:
                    if st.done.is_set() and st.q.empty():
                        break
        finally:
            with self._lock:
                self._streams.pop(rid, None)

    def cancel(self, rid: int):
        self.engine.cancel(rid)

    # --- lifecycle -----------------------------------------------------
    def drain(self, timeout: float = 30.0) -> bool:
        """Stop admitting new work and wait for in-flight requests to finish.
        Returns True if the queue emptied within `timeout`."""
        self.draining = True
        deadline = time.perf_counter() + timeout
        while self.queue_depth() > 0 and time.perf_counter() < deadline:
            time.sleep(0.05)
        return self.queue_depth() == 0

    def stop(self):
        self.engine.stop()

    def prometheus(self) -> str:
        with self.metrics.lock:
            elapsed = time.perf_counter() - self.metrics.start
            tput = self.metrics.out_tokens / elapsed if elapsed > 0 else 0.0
            ttfts = sorted(self.metrics.ttft)
            p99 = ttfts[min(len(ttfts) - 1, int(len(ttfts) * 0.99))] * 1e3 if ttfts else 0.0
            m = self.metrics
            snap = (m.accepted, m.completed, m.shed, m.cancelled, m.timed_out, m.out_tokens)
        acc, comp, shed, canc, tmo, out = snap
        rows = [
            ("nanoserve_requests_accepted_total", acc),
            ("nanoserve_requests_completed_total", comp),
            ("nanoserve_requests_shed_total", shed),
            ("nanoserve_requests_cancelled_total", canc),
            ("nanoserve_requests_timed_out_total", tmo),
            ("nanoserve_queue_depth", self.queue_depth()),
            ("nanoserve_draining", int(self.draining)),
            ("nanoserve_output_tokens_total", out),
            ("nanoserve_throughput_tokens_per_second", round(tput, 1)),
            ("nanoserve_ttft_p99_ms", round(p99, 1)),
        ]
        return "".join(f"{k} {v}\n" for k, v in rows)


def make_handler(service: InferenceService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):  # keep stdout to our own structured logs
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
                if service.draining:
                    self._send(503, "draining\n", "text/plain")
                else:
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
            prompt = body.get("prompt", "")
            max_tokens = int(body.get("max_tokens", 64))
            if body.get("stream"):
                self._stream(prompt, max_tokens)
            else:
                out = service.generate(prompt, max_tokens)
                if out is None:
                    self._send(503, '{"error":"overloaded; retry later"}')
                else:
                    self._send(200, json.dumps(out))

        def _stream(self, prompt, max_tokens):
            rid, st = service.begin_stream(prompt, max_tokens)
            if rid is None:
                self._send(503, '{"error":"overloaded; retry later"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for piece in service.stream_tokens(rid, st):
                    self.wfile.write(f"data: {json.dumps({'token': piece})}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(f"data: {json.dumps({'done': True, 'status': st.status})}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # client hung up -> cancel; the engine reclaims its KV blocks
                service.cancel(rid)

    return Handler
