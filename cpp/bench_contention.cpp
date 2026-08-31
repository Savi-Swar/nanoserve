// Freelist contention: mutex vs Treiber under 1..N threads.
//
// Each thread runs alloc/free_block pairs against one shared freelist. The
// pool is big enough that alloc never fails, so every op is the fast path and
// the number measures synchronization cost, not exhaustion handling. Median
// of 5 repeats per cell; a compiler barrier keeps the popped index live so
// the loop can't be optimized away. No frequency pinning here (macOS has no
// public knob for it), which is exactly why we take medians and report the
// crossover, not the absolute single-thread number.
//
//   make contention && ./bench_contention [max_threads]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "allocator.hpp"

using clk = std::chrono::steady_clock;

// The naive CAS loop, no backoff: what nano::TreiberFreeList was before the
// numbers from this bench promoted exponential backoff into it. Kept here so
// the collapse it suffers under contention stays reproducible.
class NaiveTreiber final : public nano::FreeList {
    static uint64_t pack(uint32_t idx, uint32_t tag) {
        return (uint64_t(tag) << 32) | idx;
    }
    std::vector<std::atomic<uint32_t>> next_;
    std::atomic<uint64_t> head_;
public:
    explicit NaiveTreiber(uint32_t n) : next_(n) {
        for (uint32_t i = 0; i < n; ++i)
            next_[i].store(i + 1 == n ? nano::kNull : i + 1,
                           std::memory_order_relaxed);
        head_.store(pack(n ? 0 : nano::kNull, 0), std::memory_order_relaxed);
    }
    uint32_t alloc() override {
        uint64_t h = head_.load(std::memory_order_acquire);
        for (;;) {
            uint32_t idx = static_cast<uint32_t>(h);
            if (idx == nano::kNull) return nano::kNull;
            uint32_t nxt = next_[idx].load(std::memory_order_relaxed);
            uint64_t nh = pack(nxt, static_cast<uint32_t>(h >> 32) + 1);
            if (head_.compare_exchange_weak(h, nh, std::memory_order_acq_rel,
                                            std::memory_order_acquire))
                return idx;
        }
    }
    void free_block(uint32_t idx) override {
        uint64_t h = head_.load(std::memory_order_relaxed);
        for (;;) {
            next_[idx].store(static_cast<uint32_t>(h),
                             std::memory_order_relaxed);
            uint64_t nh = pack(idx, static_cast<uint32_t>(h >> 32) + 1);
            if (head_.compare_exchange_weak(h, nh, std::memory_order_release,
                                            std::memory_order_relaxed))
                return;
        }
    }
    uint32_t num_free() const override { return 0; }  // not used by the bench
};

static inline void keep(uint32_t v) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(v) : "memory");
#else
    volatile uint32_t sink = v; (void)sink;
#endif
}

static double run_once(nano::FreeList& fl, int threads, uint64_t pairs) {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> ts;
    ts.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        ts.emplace_back([&] {
            ready.fetch_add(1);
            while (!go.load(std::memory_order_acquire)) {}
            for (uint64_t i = 0; i < pairs; ++i) {
                uint32_t b = fl.alloc();
                keep(b);
                fl.free_block(b);
            }
        });
    }
    while (ready.load() != threads) {}
    auto t0 = clk::now();
    go.store(true, std::memory_order_release);
    for (auto& t : ts) t.join();
    double s = std::chrono::duration<double>(clk::now() - t0).count();
    return double(threads) * double(pairs) * 2.0 / s;  // ops/s
}

static double median5(nano::FreeList& fl, int threads, uint64_t pairs) {
    std::vector<double> r;
    for (int i = 0; i < 5; ++i) r.push_back(run_once(fl, threads, pairs));
    std::sort(r.begin(), r.end());
    return r[2];
}

int main(int argc, char** argv) {
    int max_t = argc > 1 ? std::atoi(argv[1]) : 8;
    const uint32_t POOL = 1 << 16;
    const uint64_t PAIRS = 2'000'000;

    std::printf("shared freelist, alloc+free pairs per thread: %llu, "
                "median of 5\n", (unsigned long long)PAIRS);
    std::printf("%8s %14s %15s %16s\n", "threads", "mutex Mops/s",
                "naive-cas Mops/s", "treiber Mops/s");
    for (int t = 1; t <= max_t; t *= 2) {
        nano::MutexFreeList mu(POOL);
        NaiveTreiber nv(POOL);
        nano::TreiberFreeList lf(POOL);
        double m = median5(mu, t, PAIRS / t);
        double n = median5(nv, t, PAIRS / t);
        double b = median5(lf, t, PAIRS / t);
        std::printf("%8d %14.1f %15.1f %16.1f\n",
                    t, m / 1e6, n / 1e6, b / 1e6);
    }
    std::printf("\nnote: total work is fixed (pairs/thread scales down), so a "
                "flat line is\nperfect scaling of the fixed workload and a "
                "dropping line is contention cost.\n");
    return 0;
}
