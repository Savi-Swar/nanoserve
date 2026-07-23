"""One-shot GPU run: everything that was deferred from the CPU dev box, on CUDA.
Produces the headline graphs, JSON, and a consolidated results/summary.txt.

Run on any CUDA box (Colab T4, a rented A10/L4, etc.):

    pip install -r requirements.txt && pip install vllm   # vllm optional
    python scripts/gpu_run.py

Sized to finish in ~20-25 min on a free T4. Every step is guarded AND has a hard
timeout, so a slow/hung step (naive under load is the usual culprit) can't stall
the whole run -- it's marked failed and the rest continues. Bump --n / --rates on
a dedicated box for bigger numbers.
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


def step(title, args, timeout=STEP_TIMEOUT):
    print("\n" + "=" * 70)
    print(f">>> {title}")
    print("=" * 70)
    try:
        subprocess.run([PY, "-m", *args], check=True, timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        print(f"[!] step timed out after {timeout}s -- skipping, continuing")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[!] step failed ({e}) -- continuing")
        return False


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def write_summary(ok):
    """Consolidate the JSON outputs into results/summary.txt so the headline
    numbers are readable in one place (and easy to lift into the writeup)."""
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
        for e in ("naive", "static", "continuous", "paged"):
            if e in best:
                r = best[e]
                lines.append(f"  {e:<12} {r['throughput']:8.1f} tok/s   "
                             f"TTFT p99 {r['ttft']['p99']*1e3:8.0f} ms   "
                             f"(rate {r['rate']})")
        if "naive" in best and best["naive"]["throughput"] > 0:
            base = best["naive"]["throughput"]
            for e in ("continuous", "paged"):
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
        tag = "  (CPU smoke test — not valid)" if cx.get("cpu_smoke_test") else ""
        lines.append(f"roofline crossover (S={cx.get('seq_len')}): predicted B*="
                     f"{cx.get('predicted_crossover_batch', 0):.0f}, measured ~= "
                     f"{cx.get('measured_crossover_batch')}{tag}")
        lines.append("")

    sb = _load("results/spec_batched.json")
    if sb and sb.get("workloads"):
        for name, w in sb["workloads"].items():
            xo = w.get("measured_crossover_batch")
            peak = max((r["ratio"] for r in w["rows"]), default=0)
            tail = f"net LOSS at batch>={xo}" if xo else "stayed a win"
            lines.append(f"batched spec, {name}: peak {peak:.1f}x vs continuous, {tail}")
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


def main():
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                        "--format=csv"], check=False)
    else:
        print("[!] no nvidia-smi -- is this a CUDA box? vLLM will fail, util will be n/a.")

    ok = {}
    # 1. throughput ladder (fp16). Small n + few rates so naive (serial, and the
    # open-loop queue backs up under load) can't blow up the wall clock.
    ok["sweep"] = step("engine x rate sweep (fp16)", [
        "bench.sweep", "--engines", "naive", "static", "continuous", "paged",
        "--rates", "4", "8", "16", "--n", "32", "--max-tokens", "48",
        "--device", DEV])
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

    # 4c. does speculative decoding survive batching? (the headline)
    ok["spec_cost"] = step("spec cost model (predicted win->loss crossover)",
                           ["bench.spec_cost"])
    ok["spec_batched"] = step("batched speculative decoding: measured vs continuous", [
        "bench.spec_batched_study", "--device", DEV, "--batches", "1", "2", "4",
        "8", "16", "32", "--steps", "16"], timeout=900)

    # 5. low-noise paged-vs-continuous (the comparison CPU noise couldn't resolve)
    ok["noise"] = step("noise-floor: continuous vs paged (5 runs)", [
        "bench.repeat", "--compare", "continuous", "paged", "--runs", "5",
        "--rate", "16", "--n", "48", "--device", DEV])

    # 6. vLLM reference ceiling (needs vLLM; hard timeout since init can hang)
    ok["vllm"] = step("vLLM reference ceiling", [
        "bench.vllm_ref", "--n", "48", "--rate", "16", "--out", "results/vllm.json"],
        timeout=600)

    # 7. roofline overlay (T4 presets; override for your GPU)
    ok["roofline"] = step("roofline: predicted vs measured", [
        "bench.roofline", "--mem-bandwidth-gbps", "320", "--peak-tflops", "65",
        "--measured", "results/sweep.json"])

    # 8. roofline crossover: does the predicted crossover batch match measurement?
    ok["crossover"] = step("roofline crossover (predicted vs measured batch)", [
        "bench.crossover_study", "--device", DEV, "--batches", "1", "4", "8", "16",
        "32", "64", "--seq-len", "2048", "--steps", "12", "--mem-bandwidth-gbps", "320"],
        timeout=600)

    print("\n" + "#" * 70)
    write_summary(ok)
    print("#" * 70)
    print("graphs + JSON in results/. Headline numbers in results/summary.txt.")


if __name__ == "__main__":
    main()
