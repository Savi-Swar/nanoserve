# nanoserve — one-command reproducibility.
# POSIX-make compatible (works with macOS /usr/bin/make and GNU make).
#
#   make            # this help
#   make all        # regenerate every artifact + graph (memory, bench, plot)
#   make bench DEVICE=cuda   # run the sweep on a GPU instead of CPU

DEVICE ?= cpu

.PHONY: help install test test-all bench plot memory trace roofline crossover scale spec spec-batched prefix kvquant goodput noise audit all clean

help:
	@echo "nanoserve targets:"
	@echo "  install    pip install -r requirements.txt"
	@echo "  test       fast test suite (python -m pytest -q)"
	@echo "  test-all   full suite incl. slow equivalence oracle (RUN_SLOW=1)"
	@echo "  bench      engine x rate sweep -> results/sweep.json  (DEVICE=$(DEVICE))"
	@echo "  plot       results/sweep.json -> results/*.png"
	@echo "  memory     KV fragmentation ablation (no model) -> results/memory.json"
	@echo "  trace      replay the real Azure trace, all engines -> results/trace.json"
	@echo "  roofline   analytical throughput ceiling (no model)"
	@echo "  crossover  measured vs predicted decode crossover batch -> results/crossover.json"
	@echo "  scale      rerun the audit across model sizes (0.5B/1.5B/3B) -> results/scale.json"
	@echo "  spec       audit row 1: speculative decoding tokens/forward -> results/spec.json"
	@echo "  spec-batched  spec INSIDE the batch: measured win->loss crossover vs continuous"
	@echo "  prefix     audit row 2: prefix caching prefill savings -> results/prefix.json"
	@echo "  kvquant    audit row 3: KV quantization memory/quality -> results/kv_quant.json"
	@echo "  goodput    req/s meeting a TTFT+TPOT SLO, per engine -> results/goodput.json"
	@echo "  audit      run the whole audit (spec + prefix + kvquant)"
	@echo "  noise      noise-floor compare of two engines (N runs, 95% CI)"
	@echo "  all        memory + bench + plot (every artifact in one command)"
	@echo "  clean      remove generated results + caches"
	@echo ""
	@echo "Override the bench device with:  make bench DEVICE=cuda"

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

test-all:
	RUN_SLOW=1 python -m pytest -q

bench:
	python -m bench.sweep --engines naive static continuous paged --rates 4 8 16 --n 64 --device $(DEVICE)

plot:
	python -m bench.plot

memory:
	python -m bench.memory_study --n 64 --block-size 16

trace:
	python -m bench.trace_compare --device $(DEVICE) --n 32 --len-scale 16

roofline:
	python -m bench.roofline

crossover:
	python -m bench.crossover_study --device $(DEVICE) --batches 1 4 8 16 32 64 --seq-len 2048 --steps 12

scale:
	python -m bench.scale_study --device $(DEVICE)

spec:
	python -m bench.spec_study

spec-batched:
	python -m bench.spec_cost
	python -m bench.spec_batched_study --device $(DEVICE) --batches 1 2 4 8 16 32

prefix:
	python -m bench.prefix_study

kvquant:
	python -m bench.kv_quant_study

goodput:
	python -m bench.goodput_study --rates 2 4 8 16 --n 48 --device $(DEVICE)

audit: spec prefix kvquant

noise:
	python -m bench.repeat --compare continuous paged --runs 5 --rate 10 --n 24 --device $(DEVICE)

all: memory bench plot

clean:
	rm -f results/*.png results/*.json
	rm -rf __pycache__ .pytest_cache
