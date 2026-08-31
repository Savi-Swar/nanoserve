// Continuous-batching bookkeeping in C++, one Python crossing per step.
//
// The per-op measurement settled the design: the C++ allocator called through
// pybind11 per operation is SLOWER than Python (the crossing costs more than
// the work); batched into one call it is ~3x. So the scheduler exposes exactly
// two hot-path entry points per decode step:
//
//   plan_step()          -> (write slot, current length) per row
//   commit_step(eos_hit) -> finished row indices; frees their blocks, compacts
//
// Semantics mirror server/paged_exec.PagedBatchState + the engine's evict-
// promptly contract, and admission mirrors PagedContinuousEngine (reserve the
// whole prompt+max_new span; refuse when the pool can't hold it). The
// deterministic-replay harness runs this and the Python bookkeeping on the
// same trace and asserts identical decision-log hashes.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "allocator.hpp"

namespace nano {

class Scheduler {
    BlockAllocator alloc_;
    uint32_t block_size_;

    struct Row {
        int64_t rid;
        int64_t sid;
        uint64_t len;       // tokens whose KV is written (prompt, then +1/step)
        uint64_t emitted;   // sampled tokens so far (1 after prefill)
        uint64_t max_new;
    };
    std::vector<Row> rows_;
    int64_t next_sid_ = 0;

public:
    Scheduler(uint32_t num_blocks, uint32_t block_size, bool lockfree = false)
        : alloc_(num_blocks, block_size, lockfree), block_size_(block_size) {}

    size_t size() const { return rows_.size(); }
    uint32_t num_free_blocks() const { return alloc_.num_free(); }

    bool can_admit(uint64_t prompt_len, uint64_t max_new) const {
        return alloc_.can_admit(prompt_len + max_new);
    }

    // reserves the full span; returns the sequence id (block table via table())
    int64_t admit(int64_t rid, uint64_t prompt_len, uint64_t max_new) {
        int64_t sid = next_sid_++;
        alloc_.add_seq(sid, prompt_len + max_new);
        rows_.push_back(Row{rid, sid, prompt_len, 1, max_new});
        return sid;
    }

    const std::vector<uint32_t>& table(int64_t sid) const {
        return alloc_.table(sid);
    }

    std::vector<int64_t> row_ids() const {
        std::vector<int64_t> out;
        out.reserve(rows_.size());
        for (const Row& r : rows_) out.push_back(r.rid);
        return out;
    }

    // (slots, lens): where this step's token lands per row, and the current
    // written length per row (the kernel's lens argument, pre-increment)
    std::pair<std::vector<int64_t>, std::vector<int64_t>> plan_step() const {
        std::vector<int64_t> slots, lens;
        slots.reserve(rows_.size());
        lens.reserve(rows_.size());
        for (const Row& r : rows_) {
            const auto& t = alloc_.table(r.sid);
            slots.push_back(int64_t(t[r.len / block_size_]) * block_size_
                            + int64_t(r.len % block_size_));
            lens.push_back(int64_t(r.len));
        }
        return {std::move(slots), std::move(lens)};
    }

    // advance every row one token; eos_hit (may be empty = ignore_eos) marks
    // rows whose sampled token was EOS. Finished rows are freed and removed
    // (the engine's evict-promptly contract), surviving rows keep order.
    std::vector<int64_t> commit_step(const std::vector<uint8_t>& eos_hit) {
        std::vector<int64_t> finished;
        for (size_t i = 0; i < rows_.size(); ++i) {
            Row& r = rows_[i];
            r.len += 1;
            r.emitted += 1;
            bool done = r.emitted >= r.max_new
                        || (!eos_hit.empty() && eos_hit[i]);
            if (done) finished.push_back(int64_t(i));
        }
        // free + compact, preserving relative order of survivors
        if (!finished.empty()) {
            size_t w = 0, f = 0;
            for (size_t i = 0; i < rows_.size(); ++i) {
                if (f < finished.size() && int64_t(i) == finished[f]) {
                    alloc_.free_seq(rows_[i].sid);
                    ++f;
                } else {
                    rows_[w++] = rows_[i];
                }
            }
            rows_.resize(w);
        }
        return finished;
    }
};

}  // namespace nano
