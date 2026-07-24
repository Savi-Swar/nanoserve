"""Speculative decoding *inside* a continuous batch: tests whether speculation,
a batch-1 win, survives batching.

Each active row drafts its own tokens (prompt-lookup from its own context),
padded to a common draft length D. One batched forward over B*(1+D) tokens
verifies all of them; each row accepts the longest greedy-matching prefix and
commits a *different* number of tokens (1 + accepted). That raggedness is why
this is built on the paged cache: each row writes its committed tokens to its
own blocks, so rows growing by different amounts is free, which a contiguous
batch cache can't do.

Exact under greedy: a row only accepts a drafted token when it equals the target
model's own argmax at that position, and always emits the target's argmax.
Output is token-identical to naive; speculation changes the number of forward
passes, never the result. (Checked by the equivalence oracle.)

The finding this enables: at batch 1 the extra draft tokens are nearly free
(memory-bound step, spare compute) so speculation wins; as the batch grows past
the roofline crossover the step is compute-bound and the drafts cost real time,
so on low-acceptance (generic) traffic speculation becomes a net loss. Measure
tokens/forward and wall-clock vs the plain continuous engine across load.
"""
from __future__ import annotations

import torch
from transformers import DynamicCache

from .engine import ContinuousBatchEngine, _SENTINEL
from .paged_exec import PagedBatchState
from .request import Request


class SpecPagedState(PagedBatchState):
    """Paged batch state whose decode step speculates per row."""

    def __init__(self, model, num_blocks=4096, block_size=16, ngram=3, draft=8):
        super().__init__(model, num_blocks=num_blocks, block_size=block_size)
        self.ngram = ngram
        self.draft_len = draft
        self.context: list[list[int]] = []   # full token sequence per row
        # instrumentation
        self.forwards = 0
        self.committed_tokens = 0
        self.draft_proposed = 0
        self.draft_accepted = 0

    def add(self, reqs: list[Request]):
        before = self.size
        super().add(reqs)
        # rows [before, size) are new: context = prompt ids + the first sampled token
        for i in range(before, self.size):
            r = self.reqs[i]
            self.context.append(list(r.input_ids(self.m)) + list(r.output_tokens))

    def evict(self, rows):
        keep = [i for i in range(self.size) if i not in set(rows)]
        self.context = [self.context[i] for i in keep]
        super().evict(rows)

    def _lookup(self, context: list[int]) -> list[int]:
        ng = self.ngram
        if len(context) < ng:
            return []
        pat = context[-ng:]
        for i in range(len(context) - ng - 1, -1, -1):
            if context[i:i + ng] == pat:
                return context[i + ng:i + ng + self.draft_len]
        return []

    @torch.no_grad()
    def step(self) -> list[int]:
        import time
        dev = self.m.device
        B = self.size
        bs = self.block_size
        nL = self.store.n_layers
        pad_id = self.m.tokenizer.pad_token_id or self.m.eos_id or 0

        drafts = [self._lookup(self.context[i]) for i in range(B)]
        D = max((len(d) for d in drafts), default=0)
        Q = 1 + D
        self.draft_proposed += sum(len(d) for d in drafts)

        T_max = max(self.true_len)
        tables = [self.alloc.tables[self.sids[i]] for i in range(B)]
        keys, vals, ctx_mask = self.store.gather_batch(tables, self.true_len, T_max)
        cache = DynamicCache()
        for li in range(nL):
            cache.update(keys[li], vals[li], li)

        input_ids = torch.full((B, Q), pad_id, device=dev, dtype=torch.long)
        posids = torch.zeros((B, Q), device=dev, dtype=torch.long)
        newmask = torch.zeros((B, Q), device=dev, dtype=torch.long)
        for i in range(B):
            seq = [self.last_tok[i]] + drafts[i]
            for q in range(Q):
                posids[i, q] = self.true_len[i] + q       # valid RoPE positions (pads masked)
            for q, tok in enumerate(seq):
                input_ids[i, q] = tok
                newmask[i, q] = 1

        full_mask = torch.cat([ctx_mask, newmask], dim=1)
        out = self.m.model(
            input_ids=input_ids,
            attention_mask=full_mask,
            position_ids=posids,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(T_max, T_max + Q, device=dev),
        )
        self.forwards += 1
        preds = out.logits.argmax(-1)  # [B, Q]
        new_cache = out.past_key_values
        self.m.sync()
        t = time.perf_counter()

        finished = []
        for i in range(B):
            di = len(drafts[i])
            preds_i = preds[i].tolist()
            committed = [preds_i[0]]                  # model's real next token
            acc = 0
            for j in range(di):
                if drafts[i][j] == preds_i[j]:        # draft matched the model -> accept
                    committed.append(preds_i[j + 1])
                    acc += 1
                else:
                    break
            self.draft_accepted += acc

            r = self.reqs[i]
            # respect max_tokens and EOS so KV never exceeds the reserved span
            remaining = r.sampling.max_tokens - r.num_output
            take = committed[:max(0, remaining)]
            eos_cut = None
            if not r.sampling.ignore_eos:
                for k, tok in enumerate(take):
                    if tok == self.m.eos_id:
                        eos_cut = k + 1
                        break
            if eos_cut is not None:
                take = take[:eos_cut]

            n = len(take)
            if n and self.active[i]:
                k_layers = [new_cache.layers[li].keys[i, :, T_max:T_max + n, :] for li in range(nL)]
                v_layers = [new_cache.layers[li].values[i, :, T_max:T_max + n, :] for li in range(nL)]
                self.store.write_range(tables[i], self.true_len[i], k_layers, v_layers, n)
                self.true_len[i] += n
                for tok in take:
                    r.output_tokens.append(tok)
                    self.context[i].append(tok)
                self.last_tok[i] = take[-1]
                self.committed_tokens += n

            done = (r.num_output >= r.sampling.max_tokens) or (eos_cut is not None)
            if done and self.active[i]:
                r.finish_time = t
                self.active[i] = False
                finished.append(i)
        return finished

    def stats(self) -> dict:
        tpf = self.committed_tokens / self.forwards if self.forwards else 0.0
        ar = self.draft_accepted / self.draft_proposed if self.draft_proposed else 0.0
        return {"tokens_per_forward": tpf, "draft_accept_rate": ar, "forwards": self.forwards}


class BatchedSpecEngine(ContinuousBatchEngine):
    """Continuous batching + per-row speculation (paged KV). Same admit/evict
    scheduling as the continuous engine; the decode step speculates."""

    name = "spec_cont"

    def __init__(self, model, on_finish=None, max_batch=16, num_blocks=4096,
                 block_size=16, ngram=3, draft=8):
        # skip ContinuousBatchEngine.__init__ (it builds a contiguous BatchState)
        from .engine import Engine
        Engine.__init__(self, model, on_finish)
        self.max_batch = max_batch
        self.state = SpecPagedState(model, num_blocks=num_blocks, block_size=block_size,
                                    ngram=ngram, draft=draft)

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

            still = []
            for req in pending:
                room = self.state.size < self.max_batch
                fits = self.state.can_admit(req)
                force = self.state.size == 0 and not still
                if room and (fits or force):
                    import time
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


from .engine import ENGINES  # noqa: E402
ENGINES["spec_cont"] = BatchedSpecEngine
