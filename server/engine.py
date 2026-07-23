"""Serving engines. Week 1 ships the naive baseline; week 2/3 add
StaticBatchEngine and ContinuousBatchEngine behind the same interface.

All engines run a worker thread that pulls from a thread-safe queue, so the
load generator (which submits on a real wall clock) is decoupled from how the
engine chooses to schedule work. That decoupling is the whole point: the only
thing that changes between naive / static / continuous is the worker loop.
"""
from __future__ import annotations

import queue
import threading
import time

from .batched import BatchState
from .model import ModelRunner, sample
from .request import Request

_SENTINEL = object()


class Engine:
    name = "base"

    def __init__(self, model: ModelRunner, on_finish=None):
        self.model = model
        self.on_finish = on_finish or (lambda r: None)
        self._q: "queue.Queue" = queue.Queue()
        self._thread: threading.Thread | None = None

    def submit(self, req: Request):
        req.arrival_time = time.perf_counter()
        self._q.put(req)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._q.put(_SENTINEL)
        if self._thread:
            self._thread.join()

    def _run(self):
        raise NotImplementedError


class NaiveEngine(Engine):
    """One request start-to-finish, then the next. No overlap. The baseline
    every later number is measured against."""

    name = "naive"

    def _run(self):
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                return
            self._process(item)

    def _process(self, req: Request):
        req.schedule_time = time.perf_counter()
        ids = req.input_ids(self.model)
        req.prompt_len = len(ids)

        logits, kv, cur = self.model.prefill(ids)
        tok = sample(logits, req.sampling)
        self.model.sync()
        req.first_token_time = time.perf_counter()
        req.output_tokens.append(tok)

        while req.num_output < req.sampling.max_tokens:
            if not req.sampling.ignore_eos and tok == self.model.eos_id:
                break
            logits, kv, cur = self.model.decode(tok, kv, cur)
            tok = sample(logits, req.sampling)
            req.output_tokens.append(tok)

        self.model.sync()
        req.finish_time = time.perf_counter()
        self.on_finish(req)


class StaticBatchEngine(Engine):
    """Collect up to `batch_size` requests (waiting at most `max_wait`), run
    them together, and don't start the next batch until *every* sequence in
    the current one finishes. Short sequences sit in the batch burning GPU
    while the longest one finishes — the classic static-batching waste."""

    name = "static"

    def __init__(self, model, on_finish=None, batch_size: int = 8, max_wait: float = 0.05):
        super().__init__(model, on_finish)
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.state = BatchState(model)

    def _run(self):
        stop_after = False
        while True:
            first = self._q.get()
            if first is _SENTINEL:
                return
            first.schedule_time = time.perf_counter()
            batch = [first]
            deadline = time.perf_counter() + self.max_wait
            while len(batch) < self.batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self._q.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is _SENTINEL:
                    stop_after = True
                    break
                item.schedule_time = time.perf_counter()
                batch.append(item)

            self.state.add(batch)
            while self.state.any_active:
                for i in self.state.step():
                    self.on_finish(self.state.reqs[i])
            self.state.evict(list(range(self.state.size)))
            if stop_after:
                return


class ContinuousBatchEngine(Engine):
    """Iteration-level scheduling. Every decode step: evict whatever finished
    and admit whatever is waiting, up to `max_batch` concurrent sequences. A
    slot freed by a short request is filled immediately instead of idling —
    this is the jump that continuous batching buys."""

    name = "continuous"

    def __init__(self, model, on_finish=None, max_batch: int = 16):
        super().__init__(model, on_finish)
        self.max_batch = max_batch
        self.state = BatchState(model)

    def _run(self):
        stop = False
        while True:
            room = self.max_batch - self.state.size
            newcomers = []

            # nothing running and nothing queued -> block for the next arrival
            if self.state.size == 0 and self._q.empty():
                item = self._q.get()
                if item is _SENTINEL:
                    return
                item.schedule_time = time.perf_counter()
                newcomers.append(item)
                room -= 1

            while room > 0:  # drain whatever else is waiting, without blocking
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
                if item is _SENTINEL:
                    stop = True
                    break
                item.schedule_time = time.perf_counter()
                newcomers.append(item)
                room -= 1

            if newcomers:
                self.state.add(newcomers)

            if self.state.size > 0:
                finished = self.state.step()
                for i in finished:
                    self.on_finish(self.state.reqs[i])
                if finished:
                    self.state.evict(finished)

            if stop and self.state.size == 0 and self._q.empty():
                return


class PagedContinuousEngine(Engine):
    """Continuous batching whose KV lives in a paged block pool. Same admit/
    evict scheduling as ContinuousBatchEngine, but admission is gated by the
    block budget: a request waits (backpressure) until the pool can hold its
    reserved span. Under memory pressure this admits more concurrent sequences
    than a contiguous cache would — the paged throughput win, in execution."""

    name = "paged"

    def __init__(self, model, on_finish=None, max_batch: int = 16,
                 num_blocks: int = 4096, block_size: int = 16):
        super().__init__(model, on_finish)
        from .paged_exec import PagedBatchState
        self.max_batch = max_batch
        self.state = PagedBatchState(model, num_blocks=num_blocks, block_size=block_size)

    def _run(self):
        stop = False
        pending: list[Request] = []
        while True:
            if self.state.size == 0 and not pending and self._q.empty():
                item = self._q.get()
                if item is _SENTINEL:
                    return
                pending.append(item)
            while not self._q.empty() and len(pending) < self.max_batch:
                item = self._q.get_nowait()
                if item is _SENTINEL:
                    stop = True
                    break
                pending.append(item)

            # admit whatever currently fits in the block budget
            still = []
            for req in pending:
                room = self.state.size < self.max_batch
                fits = self.state.can_admit(req)
                force = self.state.size == 0 and not still  # guarantee progress
                if room and (fits or force):
                    req.schedule_time = time.perf_counter()
                    self.state.add([req])
                else:
                    still.append(req)
            pending = still

            if self.state.size > 0:
                finished = self.state.step()
                for i in finished:
                    self.on_finish(self.state.reqs[i])
                if finished:
                    self.state.evict(finished)

            if stop and self.state.size == 0 and not pending and self._q.empty():
                return


ENGINES = {
    NaiveEngine.name: NaiveEngine,
    StaticBatchEngine.name: StaticBatchEngine,
    ContinuousBatchEngine.name: ContinuousBatchEngine,
    PagedContinuousEngine.name: PagedContinuousEngine,
}
# SpeculativeEngine registers itself into ENGINES on import (see server/__init__)
