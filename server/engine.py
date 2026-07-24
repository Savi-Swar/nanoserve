"""Serving engines. Naive baseline plus StaticBatchEngine and
ContinuousBatchEngine behind the same interface.

Each engine runs a worker thread pulling from a thread-safe queue, so the load
generator (submitting on a real wall clock) is decoupled from how the engine
schedules. Only the worker loop changes between naive / static / continuous.
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

    def __init__(self, model: ModelRunner, on_finish=None, on_token=None, on_event=None):
        self.model = model
        self.on_finish = on_finish or (lambda r: None)
        # on_token(req, token) streams each generated token; None = no streaming
        # (and no per-token overhead on the benchmark path). on_event(name, req)
        # is the structured-logging hook; None = silent.
        self.on_token = on_token
        self.on_event = on_event
        self._q: "queue.Queue" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._cancelled: set[int] = set()
        self._has_deadlines = False

    def submit(self, req: Request):
        req.arrival_time = time.perf_counter()
        req.status = "queued"
        if req.deadline is not None:
            self._has_deadlines = True
        self._emit("queued", req)
        self._q.put(req)

    def cancel(self, req_id: int):
        """Ask the engine to drop a request mid-flight (e.g. the client hung
        up). Idempotent; the worker reaps it on its next iteration and returns
        its KV blocks to the free list."""
        with self._lock:
            self._cancelled.add(req_id)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._q.put(_SENTINEL)
        if self._thread:
            self._thread.join()

    def _run(self):
        raise NotImplementedError

    # --- request-lifecycle helpers (used by the continuous/paged workers) ---
    def _emit(self, event: str, req: Request):
        if self.on_event:
            self.on_event(event, req)

    def _skip_before_admit(self, req: Request, now: float) -> bool:
        """A queued request that was cancelled or expired before it ever ran:
        finalize it in place (release its waiter) and don't admit it."""
        with self._lock:
            hit = req.id in self._cancelled
            if hit:
                self._cancelled.discard(req.id)
        if hit:
            req.status = "cancelled"
        elif req.deadline is not None and now >= req.deadline:
            req.status = "timeout"
        else:
            return False
        req.finish_time = now
        self._emit(req.status, req)
        self.on_finish(req)
        return True

    def _collect_dead(self, state, now: float) -> list[int]:
        """Active rows to evict mid-batch because they were cancelled or ran
        past their deadline. Sets status/finish_time; the caller on_finish()es
        and evict()s them, which frees their KV blocks. Fast-paths to nothing
        when no request is cancelled and none carry a deadline, so the plain
        benchmark path pays no cost."""
        with self._lock:
            if not self._cancelled and not self._has_deadlines:
                return []
            cancelled = set(self._cancelled)
        dead = []
        for i in range(state.size):
            if not state.active[i]:
                continue
            r = state.reqs[i]
            if r.id in cancelled:
                r.status = "cancelled"
            elif r.deadline is not None and now >= r.deadline:
                r.status = "timeout"
            else:
                continue
            r.finish_time = now
            dead.append(i)
        if dead:
            with self._lock:
                for i in dead:
                    self._cancelled.discard(state.reqs[i].id)
        return dead


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
    the current one finishes. Short sequences sit burning GPU while the longest
    one finishes: the classic static-batching waste."""

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
    slot freed by a short request is filled immediately instead of idling,
    which is what continuous batching buys."""

    name = "continuous"

    def __init__(self, model, on_finish=None, on_token=None, on_event=None, max_batch: int = 16):
        super().__init__(model, on_finish, on_token, on_event)
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
                newcomers.append(item)
                room -= 1

            # skip anything cancelled/expired while it sat in the queue
            now = time.perf_counter()
            newcomers = [r for r in newcomers if not self._skip_before_admit(r, now)]
            for r in newcomers:
                r.schedule_time = time.perf_counter()
                r.status = "running"
                self._emit("scheduled", r)
            if newcomers:
                self.state.add(newcomers)
                for r in newcomers:
                    self._emit("first_token", r)
                    if self.on_token:
                        self.on_token(r, r.output_tokens[-1])

            # evict cancelled / timed-out sequences mid-batch, freeing their KV
            # blocks *before* we notify the caller, so a watcher that wakes on
            # completion always sees the pool already reclaimed.
            dead = self._collect_dead(self.state, time.perf_counter())
            if dead:
                dead_reqs = [self.state.reqs[i] for i in dead]
                self.state.evict(dead)
                for r in dead_reqs:
                    self._emit(r.status, r)
                    self.on_finish(r)

            if self.state.size > 0:
                finished = sorted(self.state.step())
                if self.on_token:
                    for i in range(self.state.size):
                        r = self.state.reqs[i]
                        self.on_token(r, r.output_tokens[-1])
                done_reqs = [self.state.reqs[i] for i in finished]
                if finished:
                    self.state.evict(finished)
                for r in done_reqs:
                    r.status = "done"
                    self._emit("finish", r)
                    self.on_finish(r)

            if stop and self.state.size == 0 and self._q.empty():
                return


class PagedContinuousEngine(Engine):
    """Continuous batching whose KV lives in a paged block pool. Same admit/
    evict scheduling as ContinuousBatchEngine, but admission is gated by the
    block budget: a request waits (backpressure) until the pool can hold its
    reserved span. Under memory pressure it admits more concurrent sequences
    than a contiguous cache would."""

    name = "paged"

    def __init__(self, model, on_finish=None, on_token=None, on_event=None,
                 max_batch: int = 16, num_blocks: int = 4096, block_size: int = 16):
        super().__init__(model, on_finish, on_token, on_event)
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

            # drop anything cancelled/expired before it was ever admitted
            now = time.perf_counter()
            pending = [r for r in pending if not self._skip_before_admit(r, now)]

            # admit whatever currently fits in the block budget
            still, newly = [], []
            for req in pending:
                room = self.state.size < self.max_batch
                fits = self.state.can_admit(req)
                force = self.state.size == 0 and not still  # guarantee progress
                if room and (fits or force):
                    req.schedule_time = time.perf_counter()
                    req.status = "running"
                    self._emit("scheduled", req)
                    self.state.add([req])
                    newly.append(req)
                else:
                    still.append(req)
            pending = still
            for r in newly:
                self._emit("first_token", r)
                if self.on_token:
                    self.on_token(r, r.output_tokens[-1])

            # evict cancelled / timed-out sequences mid-batch, freeing their KV
            # blocks *before* we notify the caller (watcher sees pool reclaimed).
            dead = self._collect_dead(self.state, time.perf_counter())
            if dead:
                dead_reqs = [self.state.reqs[i] for i in dead]
                self.state.evict(dead)
                for r in dead_reqs:
                    self._emit(r.status, r)
                    self.on_finish(r)

            if self.state.size > 0:
                finished = sorted(self.state.step())
                if self.on_token:
                    for i in range(self.state.size):
                        r = self.state.reqs[i]
                        self.on_token(r, r.output_tokens[-1])
                done_reqs = [self.state.reqs[i] for i in finished]
                if finished:
                    self.state.evict(finished)
                for r in done_reqs:
                    r.status = "done"
                    self._emit("finish", r)
                    self.on_finish(r)

            if stop and self.state.size == 0 and not pending and self._q.empty():
                return


ENGINES = {
    NaiveEngine.name: NaiveEngine,
    StaticBatchEngine.name: StaticBatchEngine,
    ContinuousBatchEngine.name: ContinuousBatchEngine,
    PagedContinuousEngine.name: PagedContinuousEngine,
}
# SpeculativeEngine registers itself into ENGINES on import (see server/__init__)
