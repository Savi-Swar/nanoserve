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
        def _guarded():
            try:
                self._run()
            except Exception:
                # a worker that dies silently strands every queued request
                # behind a timeout with no clue; make the crash loud
                import traceback
                print(f"[engine {self.name}] worker crashed:", flush=True)
                traceback.print_exc()

        self._thread = threading.Thread(target=_guarded, daemon=True)
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
                 max_batch: int = 16, num_blocks: int = 4096, block_size: int = 16,
                 fused: bool = False, alloc_backend: str = "py",
                 graphs: bool = False):
        super().__init__(model, on_finish, on_token, on_event)
        from .paged_exec import PagedBatchState
        self.max_batch = max_batch
        if fused:
            # decode attention reads the paged pool directly (Triton kernel on
            # CUDA); switches the model's attention implementation over and
            # compiles every split variant now, not inside a request's ITL
            from .kernels.paged_attention_triton import (
                use_triton_attention, warm_decode_kernels)
            self._prev_attn_impl = use_triton_attention(model.model)
            warm_decode_kernels(model.model, block_size=block_size)
        self.state = PagedBatchState(model, num_blocks=num_blocks,
                                     block_size=block_size, fused=fused,
                                     alloc_backend=alloc_backend, graphs=graphs,
                                     graph_buckets=[b for b in (1, 2, 4, 8, 16)
                                                    if b <= max_batch])

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
                if not fits and self.state.size == 0 and not still:
                    # the batch is empty, so the pool is fully free, and the
                    # request still does not fit: its reservation exceeds the
                    # whole pool and it can never run. Reject it now. The old
                    # behavior force-admitted it "to guarantee progress",
                    # which raised OutOfBlocks inside the engine thread and
                    # wedged every request queued behind it.
                    req.status = "rejected"
                    req.finish_time = time.perf_counter()
                    self._emit("rejected", req)
                    self.on_finish(req)
                    continue
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


class FusedPagedEngine(PagedContinuousEngine):
    """Paged engine with the no-gather decode path: Triton kernel over the
    pool on CUDA, gather fallback elsewhere. Separate name so benchmarks can
    compare it against the gather-based paged engine directly.

    Switching the attention implementation mutates the shared model, so stop()
    restores the previous implementation; otherwise every engine constructed
    after this one in the same process would silently run the kernel too."""

    name = "paged_fused"

    def __init__(self, model, on_finish=None, on_token=None, on_event=None,
                 max_batch: int = 16, num_blocks: int = 4096, block_size: int = 16,
                 alloc_backend: str = "py", graphs: bool = False):
        super().__init__(model, on_finish, on_token, on_event,
                         max_batch=max_batch, num_blocks=num_blocks,
                         block_size=block_size, fused=True,
                         alloc_backend=alloc_backend, graphs=graphs)
        self._attn_model = model.model

    def stop(self):
        super().stop()
        from .kernels.paged_attention_triton import restore_attention
        restore_attention(self._attn_model, self._prev_attn_impl)


class GraphPagedEngine(FusedPagedEngine):
    """The fused engine with the decode step captured in CUDA graphs, one
    per batch bucket. Falls back to the plain fused step off-bucket. CUDA
    only; on CPU it quietly behaves like paged_fused."""

    name = "paged_fused_graph"

    def __init__(self, model, on_finish=None, on_token=None, on_event=None,
                 max_batch: int = 16, num_blocks: int = 4096, block_size: int = 16):
        super().__init__(model, on_finish, on_token, on_event,
                         max_batch=max_batch, num_blocks=num_blocks,
                         block_size=block_size, graphs=True)


class InterleavedPagedEngine(Engine):
    """Two fused paged half-batches interleaved for a pipeline-sharded model.

    With layers split across GPUs, one batch runs the stages sequentially and
    each GPU idles while the other works (measured 36% mean utilization on
    the 7B two-T4 split). This engine keeps two half-batches at independent
    decode steps and issues their forwards back to back: CUDA queues both
    without waiting, so B's first stage runs on gpu0 while A's second stage
    runs on gpu1. Sampling, the sync point, happens after both are in
    flight. Single-GPU it degrades to two smaller batches (usually slower);
    it exists for the sharded case."""

    name = "paged_fused_pp2"

    def __init__(self, model, on_finish=None, on_token=None, on_event=None,
                 max_batch: int = 16, num_blocks: int = 4096, block_size: int = 16):
        super().__init__(model, on_finish, on_token, on_event)
        from .kernels.paged_attention_triton import (use_triton_attention,
                                                     warm_decode_kernels)
        from .paged_exec import PagedBatchState
        self.max_batch = max_batch
        self._attn_model = model.model
        self._prev_attn = use_triton_attention(model.model)
        warm_decode_kernels(model.model, block_size=block_size)
        # each half-batch gets the FULL max_batch, doubling in-flight work.
        # Decode is weight-bandwidth-bound: interleaving the same total work
        # fills the bubble but not the throughput (each GPU still streams the
        # same bytes per emitted token); two full batches at independent
        # steps let each GPU emit max_batch tokens per half-weight read.
        self._half = max_batch
        self.states = [PagedBatchState(model, num_blocks=num_blocks // 2,
                                       block_size=block_size, fused=True)
                       for _ in range(2)]
        self.state = self.states[0]   # cancel-path/tests poke .state

    def stop(self):
        super().stop()
        from .kernels.paged_attention_triton import restore_attention
        restore_attention(self._attn_model, self._prev_attn)

    def _size(self):
        return sum(st.size for st in self.states)

    def _run(self):
        stop = False
        pending: list[Request] = []
        while True:
            if self._size() == 0 and not pending and self._q.empty():
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

            now = time.perf_counter()
            pending = [r for r in pending if not self._skip_before_admit(r, now)]

            still, newly = [], []
            for req in pending:
                # emptier half first, so the two stay balanced
                st = min(self.states, key=lambda s: s.size)
                room = st.size < self._half
                fits = st.can_admit(req)
                if not fits and self._size() == 0 and not still:
                    req.status = "rejected"
                    req.finish_time = time.perf_counter()
                    self._emit("rejected", req)
                    self.on_finish(req)
                    continue
                force = self._size() == 0 and not still
                if room and (fits or force):
                    req.schedule_time = time.perf_counter()
                    req.status = "running"
                    self._emit("scheduled", req)
                    st.add([req])
                    newly.append(req)
                else:
                    still.append(req)
            pending = still
            for r in newly:
                self._emit("first_token", r)
                if self.on_token:
                    self.on_token(r, r.output_tokens[-1])

            for st in self.states:
                dead = self._collect_dead(st, time.perf_counter())
                if dead:
                    dead_reqs = [st.reqs[i] for i in dead]
                    st.evict(dead)
                    for r in dead_reqs:
                        self._emit(r.status, r)
                        self.on_finish(r)

            live = [st for st in self.states if st.size > 0]
            if live:
                # issue every forward before sampling any: the sharded
                # stages of different halves overlap across GPUs
                logits = [st.forward_async() for st in live]
                for st, lg in zip(live, logits):
                    finished = sorted(st.finish_async(lg))
                    if self.on_token:
                        for i in range(st.size):
                            self.on_token(st.reqs[i], st.reqs[i].output_tokens[-1])
                    done_reqs = [st.reqs[i] for i in finished]
                    if finished:
                        st.evict(finished)
                    for r in done_reqs:
                        r.status = "done"
                        self._emit("finish", r)
                        self.on_finish(r)

            if stop and self._size() == 0 and not pending and self._q.empty():
                return


class CppPagedEngine(FusedPagedEngine):
    """The fused paged engine with block bookkeeping in nanoserve_core (the
    C++ allocator). Serving through it is the integration proof for the C++
    pillar: the replay harness shows the decisions match by hash, this shows
    the served tokens match. Requires the extension (make cpp)."""

    name = "paged_fused_cpp"

    def __init__(self, model, on_finish=None, on_token=None, on_event=None,
                 max_batch: int = 16, num_blocks: int = 4096, block_size: int = 16):
        super().__init__(model, on_finish, on_token, on_event,
                         max_batch=max_batch, num_blocks=num_blocks,
                         block_size=block_size, alloc_backend="cpp")


class ContinuousFusedEngine(ContinuousBatchEngine):
    """Continuous engine with decode attention through the Triton kernel
    (contiguous cache; the kernel replaces repeat_kv + SDPA for q_len == 1).
    Restores the model's attention implementation on stop()."""

    name = "continuous_fused"

    def __init__(self, model, on_finish=None, on_token=None, on_event=None,
                 max_batch: int = 16):
        super().__init__(model, on_finish, on_token, on_event, max_batch=max_batch)
        from .kernels.paged_attention_triton import (use_triton_attention,
                                                     warm_decode_kernels)
        self._attn_model = model.model
        self._prev_attn = use_triton_attention(model.model)
        warm_decode_kernels(model.model)

    def stop(self):
        super().stop()
        from .kernels.paged_attention_triton import restore_attention
        restore_attention(self._attn_model, self._prev_attn)


ENGINES = {
    NaiveEngine.name: NaiveEngine,
    StaticBatchEngine.name: StaticBatchEngine,
    ContinuousBatchEngine.name: ContinuousBatchEngine,
    PagedContinuousEngine.name: PagedContinuousEngine,
    FusedPagedEngine.name: FusedPagedEngine,
    ContinuousFusedEngine.name: ContinuousFusedEngine,
    CppPagedEngine.name: CppPagedEngine,
    GraphPagedEngine.name: GraphPagedEngine,
    InterleavedPagedEngine.name: InterleavedPagedEngine,
}
# SpeculativeEngine registers itself into ENGINES on import (see server/__init__)
