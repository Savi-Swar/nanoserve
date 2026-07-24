"""Prompt-lookup speculative decoding (PLD): the first optimization under audit.
Model-free speculation: instead of a draft model, guess the next few tokens by
finding where the last n-gram of the context appeared *earlier* in the context
and proposing whatever followed it. Verify all guesses in one forward pass;
accept the longest greedy-matching prefix.

Why it's exact: under greedy we only accept a drafted token when it equals the
target model's own argmax at that position, and always emit the target's token
at the first mismatch. Output is token-identical to naive decoding; speculation
changes how many forward passes it takes, never the result. (Checked by the
equivalence oracle.)

Single-sequence by design: the batch-1 regime where the papers report 2-3x. The
audit's finding is what happens to that number under continuous batching and on
real traffic, measured separately.
"""
from __future__ import annotations

import time

from .engine import ENGINES, Engine, _SENTINEL
from .request import Request


class SpeculativeEngine(Engine):
    name = "spec"

    def __init__(self, model, on_finish=None, ngram: int = 3, draft: int = 8):
        super().__init__(model, on_finish)
        self.ngram = ngram
        self.draft = draft
        # instrumentation: tokens emitted per forward pass
        self.forwards = 0
        self.committed = 0
        self.draft_proposed = 0
        self.draft_accepted = 0

    def _run(self):
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                return
            self._process(item)

    def _lookup(self, context: list[int]) -> list[int]:
        """Most-recent earlier occurrence of the last n-gram -> its continuation."""
        ng = self.ngram
        if len(context) < ng:
            return []
        pat = context[-ng:]
        for i in range(len(context) - ng - 1, -1, -1):
            if context[i:i + ng] == pat:
                return context[i + ng:i + ng + self.draft]
        return []

    def _process(self, req: Request):
        req.schedule_time = time.perf_counter()
        m = self.model
        ids = req.input_ids(m)
        req.prompt_len = len(ids)
        max_new = req.sampling.max_tokens
        eos = m.eos_id

        logits, cache, cur = m.prefill(ids)  # cache holds context[:cur]
        self.forwards += 1
        tok = int(logits.argmax(-1))
        m.sync()
        req.first_token_time = time.perf_counter()
        req.output_tokens.append(tok)
        self.committed += 1
        context = ids + [tok]  # context[cur] = tok is committed but not yet in cache

        while req.num_output < max_new:
            if not req.sampling.ignore_eos and tok == eos:
                break
            remaining = max_new - req.num_output - 1  # -1: the guaranteed model token
            draft = self._lookup(context)[:max(0, remaining)]
            seq = [context[cur]] + draft  # feed the pending token + the guesses
            logits, cache, _ = m.decode_many(seq, cache, cur)
            self.forwards += 1
            self.draft_proposed += len(draft)

            preds = logits.argmax(-1).tolist()  # preds[i] = model token after seq[i]
            committed = [preds[0]]              # n_1: the real next token, always kept
            acc = 0
            for j in range(len(draft)):
                if draft[j] == preds[j]:        # guess matched the model -> accept
                    committed.append(preds[j + 1])
                    acc += 1
                else:
                    break
            self.draft_accepted += acc

            # keep KV for the pending token + accepted drafts; drop the rest
            cur = cur + 1 + acc
            m.crop_cache(cache, cur)

            for t in committed:
                if req.num_output >= max_new:
                    break
                req.output_tokens.append(t)
                context.append(t)
                tok = t
                self.committed += 1

        m.sync()
        req.finish_time = time.perf_counter()
        self.on_finish(req)

    def stats(self) -> dict:
        tpf = self.committed / self.forwards if self.forwards else 0.0
        ar = self.draft_accepted / self.draft_proposed if self.draft_proposed else 0.0
        return {"tokens_per_forward": tpf, "draft_accept_rate": ar,
                "forwards": self.forwards, "committed": self.committed}

    def stop(self):
        super().stop()
        s = self.stats()
        print(f"[spec] tokens/forward={s['tokens_per_forward']:.2f}  "
              f"draft_accept_rate={s['draft_accept_rate']:.2f}  "
              f"(forwards={s['forwards']}, tokens={s['committed']})")


ENGINES["spec"] = SpeculativeEngine
