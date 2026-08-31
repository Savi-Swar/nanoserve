"""One-shot GPU run: everything deferred from the CPU dev box, on CUDA.
Produces the graphs, JSON, and a consolidated results/summary.txt.

Run on any CUDA box (Colab T4, a rented A10/L4, etc.):

    pip install -r requirements.txt && pip install vllm   # vllm optional
    python scripts/gpu_run.py

Sized for ~20-25 min on a free T4. Every step is guarded with a hard timeout, so
a slow/hung step (naive under load, usually) can't stall the run: it's marked
failed and the rest continues. Bump --n / --rates on a dedicated box for bigger
numbers.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

DEV = "cuda"
PY = sys.executable
STEP_TIMEOUT = 720  # seconds; hard cap per step so nothing hangs the run


def step(title, args, timeout=STEP_TIMEOUT, env=None, tee=None):
    header = "\n" + "=" * 70 + f"\n>>> {title}\n" + "=" * 70
    print(header)
    run_env = dict(os.environ, **env) if env else None

    def _log(text):
        if tee and text:
            os.makedirs(os.path.dirname(tee) or ".", exist_ok=True)
            with open(tee, "a") as f:
                f.write(text)

    _log(header + "\n")
    try:
        r = subprocess.run([PY, "-m", *args], check=True, timeout=timeout,
                           env=run_env, capture_output=tee is not None, text=True)
        if tee is not None:
            print(r.stdout, end="")
            _log(r.stdout + (r.stderr or ""))
        return True
    except subprocess.TimeoutExpired:
        print(f"[!] step timed out after {timeout}s, skipping")
        _log(f"[!] step timed out after {timeout}s\n")
        return False
    except subprocess.CalledProcessError as e:
        if tee is not None:
            print(e.stdout or "", end="")
            print(e.stderr or "", end="")
            _log((e.stdout or "") + (e.stderr or ""))
        print(f"[!] step failed ({e}), continuing")
        _log(f"[!] step failed ({e})\n")
        return False


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def write_summary(ok):
    """Consolidate the JSON outputs into results/summary.txt: readable in one
    place, easy to lift into the writeup."""
    lines = ["nanoserve GPU run summary", "=" * 40, ""]
    lines.append("steps: " + ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in ok.items()))
    lines.append("")

    sweep = _load("results/sweep.json")
    if sweep and sweep.get("runs"):
        lines.append("throughput ladder (peak tok/s per engine, fp16):")
        best = {}
        for r in sweep["runs"]:
            e = r["engine"]
            if e not in best or r["throughput"] > best[e]["throughput"]:
                best[e] = r
        for e in ("naive", "static", "continuous", "continuous_fused",
                  "paged", "paged_fused"):
            if e in best:
                r = best[e]
                lines.append(f"  {e:<17} {r['throughput']:8.1f} tok/s   "
                             f"TTFT p99 {r['ttft']['p99']*1e3:8.0f} ms   "
                             f"(rate {r['rate']})")
        if "naive" in best and best["naive"]["throughput"] > 0:
            base = best["naive"]["throughput"]
            for e in ("continuous", "continuous_fused", "paged", "paged_fused"):
                if e in best:
                    lines.append(f"  {e} vs naive: {best[e]['throughput']/base:.1f}x")
        lines.append("")

    vllm = _load("results/vllm.json")
    if vllm:
        lines.append(f"vLLM reference: {vllm.get('throughput', '?')} tok/s "
                     f"(TTFT p99 {vllm.get('ttft', {}).get('p99', 0)*1e3:.0f} ms)")
        if sweep and sweep.get("runs"):
            best_ours = max((r["throughput"] for r in sweep["runs"]), default=0)
            if vllm.get("throughput"):
                lines.append(f"  nanoserve best is {best_ours/vllm['throughput']*100:.0f}% of vLLM")
        lines.append("")

    mem = _load("results/memory.json")
    if mem:
        s = mem.get("strategies", {})
        if s:
            lines.append(f"paged fragmentation: reserve {s['reserve_max']['frag']*100:.0f}% "
                         f"vs paged {s['paged']['frag']*100:.0f}%")
        cap = mem.get("capacity_under_budget", {})
        if cap:
            lines.append(f"  seqs in {cap.get('budget_mib')} MiB: reserve {cap.get('reserve_max')} "
                         f"vs paged {cap.get('paged')}")
        lines.append("")

    gp = _load("results/goodput.json")
    if gp and gp.get("runs"):
        caps = {e: max(x["goodput_qps"] for x in r) for e, r in gp["runs"].items()}
        lines.append(f"goodput req/s (SLO {gp['ttft_slo_ms']:.0f}ms TTFT / {gp['tpot_slo_ms']:.0f}ms TPOT): "
                     + ", ".join(f"{e} {c:.1f}" for e, c in caps.items()))
        if caps.get("naive"):
            best = max(caps, key=caps.get)
            lines.append(f"  {best} sustains {caps[best] / caps['naive']:.1f}x naive's goodput under SLO")
        lines.append("")

    cx = _load("results/crossover.json")
    if cx:
        tag = "  (CPU smoke test, not valid)" if cx.get("cpu_smoke_test") else ""
        lines.append(f"roofline crossover (S={cx.get('seq_len')}): predicted B*="
                     f"{cx.get('predicted_crossover_batch', 0):.0f}, measured ~= "
                     f"{cx.get('measured_crossover_batch')}{tag}")
        cxt = _load("results/crossover_triton.json")
        if cxt:
            lines.append(f"  with triton decode kernel: measured ~= "
                         f"{cxt.get('measured_crossover_batch')}")
            rows_s = {r['batch']: r['decode_tok_s'] for r in cx.get('sweep', [])}
            rows_t = {r['batch']: r['decode_tok_s'] for r in cxt.get('sweep', [])}
            common = sorted(set(rows_s) & set(rows_t))
            if common:
                lines.append("  decode tok/s sdpa->triton: " + "  ".join(
                    f"B{b}:{rows_s[b]:.0f}->{rows_t[b]:.0f}" for b in common))
        lines.append("")

    kb = _load("results/kernel_bench.json")
    if kb and kb.get("rows"):
        cells = "  ".join(
            f"B{r['batch']}:{r.get('split_speedup', r['speedup']):.1f}x"
            for r in kb["rows"])
        lines.append(f"decode kernel vs sdpa (T={kb.get('seq_len')}): {cells}")
        lines.append("")

    sb = _load("results/spec_batched.json")
    if sb and sb.get("workloads"):
        for name, w in sb["workloads"].items():
            xo = w.get("measured_crossover_batch")
            peak = max((r["ratio"] for r in w["rows"]), default=0)
            tail = f"net LOSS at batch>={xo}" if xo else "stayed a win"
            lines.append(f"batched spec, {name}: peak {peak:.1f}x vs continuous, {tail}")
        lines.append("")

    sc = _load("results/scale.json")
    if sc and sc.get("results"):
        lines.append("scale axis (batch-1 spec tok/forward + predicted B*):")
        for r in sc["results"]:
            name = r["model"].split("/")[-1]
            lines.append(f"  {name:<14} generic {r['spec_generic_tpf']:.2f}  "
                         f"grounded {r['spec_grounded_tpf']:.2f}  "
                         f"prefix {r['prefix_saved'] * 100:.0f}%  "
                         f"8bit-ppl-d {r['ppl_delta_8bit']:+.2f}  "
                         f"B* {r['predicted_crossover_batch']:.0f}")
        lines.append("")

    for name, path in [("spec", "results/spec.json"), ("prefix", "results/prefix.json"),
                       ("kv_quant", "results/kv_quant.json")]:
        d = _load(path)
        if d:
            lines.append(f"audit {name}: {json.dumps(d)[:300]}")
    lines.append("")

    text = "\n".join(lines)
    os.makedirs("results", exist_ok=True)
    with open("results/summary.txt", "w") as f:
        f.write(text)
    print("\n" + text)
    print("wrote results/summary.txt")


def read_mode() -> str:
    """The Kaggle notebook always runs this script verbatim, so the committed
    mode file is how a push selects what the next headless run does.
    'full' = the whole pipeline; 'kernel' = Triton kernel tests + microbench
    only (fast iteration loop for kernel work)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_run_mode.txt")
    try:
        with open(path) as f:
            return f.read().strip() or "full"
    except OSError:
        return "full"


def kernel_run():
    import torch
    import transformers
    print(f"torch {torch.__version__}  transformers {transformers.__version__}")
    try:
        import triton
        print(f"triton {triton.__version__}")
    except ImportError:
        print("[!] triton not importable")
    LOG = "results/kernel_ci_log.txt"
    ok = {}
    ok["kernel_tests"] = step("triton kernel equivalence tests",
                              ["pytest", "tests/test_kernel_equivalence.py", "-q",
                               "-rf", "--tb=short"], tee=LOG)
    ok["model_diag"] = step("kernel vs sdpa model-level diagnosis",
                            ["bench.kernel_model_diag", "--device", "cuda"], tee=LOG)
    ok["fused_exact"] = step("fused paged path token-exactness",
                             ["pytest", "tests/test_fused_paged.py", "-q", "--tb=short"],
                             env={"RUN_SLOW": "1"}, tee=LOG)
    ok["kernel_bench"] = step("kernel microbench", ["bench.kernel_bench"], tee=LOG)
    ok["overhead_gate"] = step("phase-2 gate: python overhead share of a step",
                               ["bench.overhead_gate", "--device", "cuda"], tee=LOG)
    # end-to-end: the same open-loop workload through gather-paged vs fused-paged
    ok["e2e"] = step("engine sweep: paged vs paged_fused", [
        "bench.sweep", "--engines", "paged", "paged_fused",
        "--rates", "16", "--n", "32", "--max-tokens", "48", "--device", "cuda",
        "--out", "results/kernel_e2e.json"], tee=LOG)
    print("\nsteps: " + ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in ok.items()))
    os.makedirs("results", exist_ok=True)
    with open("results/summary.txt", "w") as f:
        f.write("kernel-mode run\n" + ", ".join(
            f"{k}={'ok' if v else 'FAIL'}" for k, v in ok.items()) + "\n")


def main():
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                        "--format=csv"], check=False)
    else:
        print("[!] no nvidia-smi; is this a CUDA box? vLLM will fail, util will be n/a.")

    for mode in read_mode().split("+"):
        run_mode(mode)


def run_mode(mode):
    if mode == "kernel":
        print(">>> mode: kernel (tests + microbench only; set "
              "scripts/gpu_run_mode.txt to 'full' for the whole pipeline)")
        kernel_run()
        return
    if mode == "tune":
        print(">>> mode: tune (kernel launch-parameter sweep)")
        step("kernel tuner", ["bench.kernel_tune"], tee="results/kernel_ci_log.txt")
        os.makedirs("results", exist_ok=True)
        with open("results/summary.txt", "w") as f:
            f.write("tune-mode run\n")
        return
    if mode == "cppstorm":
        # just the c++-engine storm leg (the full sweep already ran)
        print(">>> mode: cppstorm (build extension, storm the c++ engine)")
        LOG = "results/kernel_ci_log.txt"
        os.system(f"{PY} -m pip install -q pybind11 && make cpp")
        step("storms: paged_fused_cpp", [
            "bench.storm_study", "--engine", "paged_fused_cpp",
            "--device", "cuda", "--seeds", "0",
            "--out", "results/storm_cpp.json"], tee=LOG, timeout=900)
        os.makedirs("results", exist_ok=True)
        with open("results/summary.txt", "w") as f:
            f.write("cppstorm-mode run\n")
        return
    if mode == "graph":
        print(">>> mode: graph (CUDA-graph decode: correctness, step time, tails)")
        LOG = "results/kernel_ci_log.txt"
        step("graph equivalence tests",
             ["pytest", "tests/test_graph_step.py", "-q"], tee=LOG)
        step("graph step-time bench", ["bench.graph_bench", "--device", "cuda"],
             tee=LOG, timeout=1500)
        step("ITL tails: paged_fused vs paged_fused_graph", [
            "bench.latency_study", "--engines", "paged_fused",
            "paged_fused_graph", "--device", "cuda",
            "--out", "results/latency_graph.json"], tee=LOG, timeout=1500)
        os.makedirs("results", exist_ok=True)
        with open("results/summary.txt", "w") as f:
            f.write("graph-mode run\n")
        return
    if mode == "storm":
        print(">>> mode: storm (cancellation storms, survivor tails, invariants)")
        LOG = "results/kernel_ci_log.txt"
        step("storms: paged_fused", [
            "bench.storm_study", "--engine", "paged_fused",
            "--device", "cuda"], tee=LOG, timeout=1800)
        # the c++-backed engine under the same storms (1 seed: integration
        # proof, not a tail study). Extension built here; failure just skips.
        os.system(f"{PY} -m pip install -q pybind11 && make cpp")
        step("storms: paged_fused_cpp", [
            "bench.storm_study", "--engine", "paged_fused_cpp",
            "--device", "cuda", "--seeds", "0",
            "--out", "results/storm_cpp.json"], tee=LOG, timeout=900)
        os.makedirs("results", exist_ok=True)
        with open("results/summary.txt", "w") as f:
            f.write("storm-mode run\n")
        return
    if mode == "latency":
        print(">>> mode: latency (ITL tails, open-loop, 5 runs/engine)")
        LOG = "results/kernel_ci_log.txt"
        step("phase-2 gate: python overhead share of a step", [
            "bench.overhead_gate", "--device", "cuda"], tee=LOG)
        step("ITL tails: paged vs paged_fused", [
            "bench.latency_study", "--engines", "paged", "paged_fused",
            "--device", "cuda"], tee=LOG, timeout=1500)
        step("ITL tails: continuous vs continuous_fused", [
            "bench.latency_study", "--engines", "continuous", "continuous_fused",
            "--device", "cuda", "--out", "results/latency_cont.json"],
            tee=LOG, timeout=1500)
        os.makedirs("results", exist_ok=True)
        with open("results/summary.txt", "w") as f:
            f.write("latency-mode run\n")
        return

    ok = {}
    # 1. throughput ladder (fp16). Small n + few rates so naive (serial, and the
    # open-loop queue backs up under load) can't blow up the wall clock.
    ok["sweep"] = step("engine x rate sweep (fp16)", [
        "bench.sweep", "--engines", "naive", "static", "continuous",
        "continuous_fused", "paged", "paged_fused",
        "--rates", "4", "8", "16", "--n", "32", "--max-tokens", "48",
        "--device", DEV], timeout=900)
    ok["plot"] = step("plots", ["bench.plot"])

    # 2. deterministic memory ablation (no model)
    ok["memory"] = step("KV fragmentation ablation", ["bench.memory_study", "--n", "128"])

    # 3. real Azure trace. len-scale 4 keeps contexts/gens tractable for the
    # serial engines while preserving the heavy-tailed shape.
    ok["trace"] = step("Azure trace", [
        "bench.trace_compare", "--device", DEV, "--n", "32", "--len-scale", "4"])

    # 4. audit rows at fp16
    ok["spec"] = step("audit: speculative decoding", ["bench.spec_study", "--device", DEV])
    ok["prefix"] = step("audit: prefix caching", ["bench.prefix_study", "--device", DEV])
    ok["kvquant"] = step("audit: KV quantization", ["bench.kv_quant_study", "--device", DEV])

    # 4b. goodput under an SLO (req/s meeting both TTFT and TPOT targets)
    ok["goodput"] = step("goodput under SLO (500ms TTFT / 50ms TPOT)", [
        "bench.goodput_study", "--engines", "naive", "static", "continuous", "paged",
        "--rates", "4", "8", "16", "--n", "32", "--max-tokens", "48",
        "--ttft-slo", "500", "--tpot-slo", "50", "--device", DEV])

    # 4c. does speculative decoding survive batching?
    ok["spec_cost"] = step("spec cost model (predicted win->loss crossover)",
                           ["bench.spec_cost"])
    ok["spec_batched"] = step("batched speculative decoding: measured vs continuous", [
        "bench.spec_batched_study", "--device", DEV, "--batches", "1", "2", "4",
        "8", "16", "32", "--steps", "16"], timeout=900)

    # 5. low-noise paged-vs-continuous (the comparison CPU noise couldn't resolve)
    ok["noise"] = step("noise-floor: continuous vs paged (5 runs)", [
        "bench.repeat", "--compare", "continuous", "paged", "--runs", "5",
        "--rate", "16", "--n", "48", "--device", DEV])

    # 5b. does the kernel move the engine? same noise-floor discipline
    ok["noise_kernel"] = step("noise-floor: paged vs paged_fused (5 runs)", [
        "bench.repeat", "--compare", "paged", "paged_fused", "--runs", "5",
        "--rate", "16", "--n", "48", "--device", DEV])

    # 6. roofline crossover: does the predicted crossover batch match measurement?
    # (run BEFORE vLLM; vLLM's EngineCore lingers on GPU memory and would OOM this)
    ok["crossover"] = step("roofline crossover (predicted vs measured batch)", [
        "bench.crossover_study", "--device", DEV, "--batches", "1", "4", "8", "16",
        "32", "64", "--seq-len", "2048", "--steps", "12", "--mem-bandwidth-gbps", "320"],
        timeout=600)

    # 6b. same crossover through the triton kernel: does the knee move right?
    ok["crossover_triton"] = step("crossover with the triton decode kernel", [
        "bench.crossover_study", "--device", DEV, "--batches", "1", "4", "8", "16",
        "32", "64", "--seq-len", "2048", "--steps", "12", "--mem-bandwidth-gbps", "320",
        "--attn", "triton", "--out", "results/crossover_triton.json"],
        timeout=600)

    # 6c. op-level kernel table for the writeup
    ok["kernel_bench"] = step("kernel microbench", ["bench.kernel_bench"])

    # 7. roofline overlay (analytical; T4 presets; override for your GPU)
    ok["roofline"] = step("roofline: predicted vs measured", [
        "bench.roofline", "--mem-bandwidth-gbps", "320", "--peak-tflops", "65",
        "--measured", "results/sweep.json"])

    # 7b. scale axis: rerun the audit slice at 1.5B and 3B (both fit a T4). Light
    # (batch-1 forwards), and it validates the moving-crossover prediction. Runs
    # BEFORE vLLM, which leaks GPU memory and would OOM the 3B load.
    ok["scale"] = step("scale axis: audit at 0.5B / 1.5B / 3B", [
        "bench.scale_study", "--device", DEV,
        "--models", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"],
        timeout=1800)

    # 8. vLLM reference ceiling LAST; its EngineCore can hold GPU memory after it
    ok["vllm"] = step("vLLM reference ceiling", [
        "bench.vllm_ref", "--n", "48", "--rate", "16", "--out", "results/vllm.json"],
        timeout=600)

    print("\n" + "#" * 70)
    write_summary(ok)
    print("#" * 70)
    print("graphs + JSON in results/. Headline numbers in results/summary.txt.")


if __name__ == "__main__":
    main()
