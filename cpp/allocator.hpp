// Fixed-pool KV block allocator, C++ port of server/paged_cache.py.
//
// Two freelist implementations behind one interface:
//   MutexFreeList    - std::vector as a LIFO stack under a mutex. The default:
//                      an uncontended lock is ~20ns and the vector's cache
//                      behavior is unbeatable single-threaded.
//   TreiberFreeList  - lock-free index stack, ABA-proofed by packing a 32-bit
//                      generation tag with the 32-bit head index in one
//                      atomic<uint64_t>. Earns its keep only if allocation
//                      ever moves off the scheduler thread; the contention
//                      benchmark decides.
//
// Both pop in the same order as the Python allocator (LIFO over the same
// initial ordering), so a single-threaded run produces bit-identical block
// tables to the Python implementation - which is what the differential test
// asserts.
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace nano {

constexpr uint32_t kNull = 0xFFFFFFFFu;

class FreeList {
public:
    virtual ~FreeList() = default;
    virtual uint32_t alloc() = 0;          // kNull when exhausted
    virtual void free_block(uint32_t idx) = 0;
    virtual uint32_t num_free() const = 0;
};

class MutexFreeList final : public FreeList {
    mutable std::mutex mu_;
    std::vector<uint32_t> free_;
public:
    explicit MutexFreeList(uint32_t n) {
        free_.reserve(n);
        // python: list(range(num_blocks - 1, -1, -1)) used as a stack ->
        // pop() returns 0, 1, 2, ... in order
        for (uint32_t i = n; i-- > 0;) free_.push_back(i);
    }
    uint32_t alloc() override {
        std::lock_guard<std::mutex> g(mu_);
        if (free_.empty()) return kNull;
        uint32_t b = free_.back();
        free_.pop_back();
        return b;
    }
    void free_block(uint32_t idx) override {
        std::lock_guard<std::mutex> g(mu_);
        free_.push_back(idx);
    }
    uint32_t num_free() const override {
        std::lock_guard<std::mutex> g(mu_);
        return static_cast<uint32_t>(free_.size());
    }
};

class TreiberFreeList final : public FreeList {
    static uint64_t pack(uint32_t idx, uint32_t tag) {
        return (uint64_t(tag) << 32) | idx;
    }
    // exponential backoff on CAS failure. The contention bench made the case:
    // without it the naive CAS loop collapses ~100x at 8 threads (every retry
    // re-hammers the one head cache line); with it the stack beats the mutex
    // at every contended thread count and costs nothing uncontended, because
    // an uncontended CAS never fails and never reaches the pause.
    static inline void backoff(uint32_t& delay) {
        for (uint32_t i = 0; i < delay; ++i) {
#if defined(__x86_64__)
            asm volatile("pause");
#elif defined(__aarch64__)
            asm volatile("isb");
#endif
        }
        if (delay < 1024) delay <<= 1;
    }
    std::vector<std::atomic<uint32_t>> next_;
    std::atomic<uint64_t> head_;
public:
    explicit TreiberFreeList(uint32_t n) : next_(n) {
        // same pop order as the mutex/python version: head starts at 0,
        // 0 -> 1 -> 2 -> ... -> null
        for (uint32_t i = 0; i < n; ++i)
            next_[i].store(i + 1 == n ? kNull : i + 1, std::memory_order_relaxed);
        head_.store(pack(n ? 0 : kNull, 0), std::memory_order_relaxed);
        static_assert(std::atomic<uint64_t>::is_always_lock_free);
    }
    uint32_t alloc() override {
        uint64_t h = head_.load(std::memory_order_acquire);
        uint32_t delay = 1;
        for (;;) {
            uint32_t idx = static_cast<uint32_t>(h);
            if (idx == kNull) return kNull;
            uint32_t nxt = next_[idx].load(std::memory_order_relaxed);
            uint64_t nh = pack(nxt, static_cast<uint32_t>(h >> 32) + 1);
            if (head_.compare_exchange_weak(h, nh, std::memory_order_acq_rel,
                                            std::memory_order_acquire))
                return idx;
            backoff(delay);
        }
    }
    void free_block(uint32_t idx) override {
        uint64_t h = head_.load(std::memory_order_relaxed);
        uint32_t delay = 1;
        for (;;) {
            next_[idx].store(static_cast<uint32_t>(h), std::memory_order_relaxed);
            uint64_t nh = pack(idx, static_cast<uint32_t>(h >> 32) + 1);
            if (head_.compare_exchange_weak(h, nh, std::memory_order_release,
                                            std::memory_order_relaxed))
                return;
            backoff(delay);
        }
    }
    uint32_t num_free() const override {
        // O(n) walk; diagnostics only, not a hot-path call
        uint32_t n = 0;
        uint64_t h = head_.load(std::memory_order_acquire);
        uint32_t idx = static_cast<uint32_t>(h);
        while (idx != kNull) {
            ++n;
            idx = next_[idx].load(std::memory_order_relaxed);
        }
        return n;
    }
};

struct SeqInfo {
    std::vector<uint32_t> table;
    uint64_t length = 0;   // tokens stored
    uint32_t fill = 0;     // tokens in the last block
};

class BlockAllocator {
    uint32_t num_blocks_;
    uint32_t block_size_;
    std::unique_ptr<FreeList> free_;
    std::unordered_map<int64_t, SeqInfo> seqs_;

public:
    BlockAllocator(uint32_t num_blocks, uint32_t block_size, bool lockfree)
        : num_blocks_(num_blocks), block_size_(block_size) {
        if (lockfree)
            free_ = std::make_unique<TreiberFreeList>(num_blocks);
        else
            free_ = std::make_unique<MutexFreeList>(num_blocks);
    }

    uint32_t num_blocks() const { return num_blocks_; }
    uint32_t block_size() const { return block_size_; }
    uint32_t num_free() const { return free_->num_free(); }
    uint32_t num_used() const { return num_blocks_ - free_->num_free(); }

    static uint32_t blocks_for(uint64_t n_tokens, uint32_t bs) {
        return n_tokens == 0 ? 1 : static_cast<uint32_t>((n_tokens + bs - 1) / bs);
    }
    bool can_admit(uint64_t n_tokens) const {
        return blocks_for(n_tokens, block_size_) <= free_->num_free();
    }

    std::vector<uint32_t> add_seq(int64_t sid, uint64_t n_tokens) {
        uint32_t nb = blocks_for(n_tokens, block_size_);
        std::vector<uint32_t> table;
        table.reserve(nb);
        for (uint32_t i = 0; i < nb; ++i) {
            uint32_t b = free_->alloc();
            if (b == kNull) {                       // roll back partial grab
                for (uint32_t x : table) free_->free_block(x);
                throw std::runtime_error("OutOfBlocks");
            }
            table.push_back(b);
        }
        SeqInfo info;
        info.table = table;
        info.length = n_tokens;
        info.fill = static_cast<uint32_t>(n_tokens - uint64_t(nb - 1) * block_size_);
        seqs_[sid] = std::move(info);
        return table;
    }

    // grows one token; returns the new block id, or kNull if none was needed
    uint32_t append_token(int64_t sid) {
        SeqInfo& s = seqs_.at(sid);
        uint32_t nb = kNull;
        if (s.fill == block_size_) {
            nb = free_->alloc();
            if (nb == kNull) throw std::runtime_error("OutOfBlocks");
            s.table.push_back(nb);
            s.fill = 0;
        }
        s.fill += 1;
        s.length += 1;
        return nb;
    }

    void free_seq(int64_t sid) {
        auto it = seqs_.find(sid);
        if (it == seqs_.end()) return;
        for (uint32_t b : it->second.table) free_->free_block(b);
        seqs_.erase(it);
    }

    const std::vector<uint32_t>& table(int64_t sid) const {
        return seqs_.at(sid).table;
    }
    uint64_t seq_length(int64_t sid) const { return seqs_.at(sid).length; }
    size_t num_seqs() const { return seqs_.size(); }
};

}  // namespace nano
