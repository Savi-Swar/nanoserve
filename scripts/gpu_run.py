"""One-shot GPU run: everything that was deferred from the CPU dev box, at real
scale on CUDA. Produces the headline graphs + numbers and a summary.

Run on any CUDA box (Colab T4, a rented A10/L4, etc.):

    pip install -r requirements.txt && pip install vllm   # vllm optional
    python scripts/gpu_run.py

Each step is guarded — if one fails (e.g. vLLM not installed) the rest still
run. Results land in results/ (JSON + PNGs). Nothing here is CPU-specific; the
same commands are what `make bench DEVICE=cuda` etc. call, just batched together
at GPU scale.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

DEV = "cuda"
PY = sys.executable
STEPS = []


def step(title, args):
    STEPS.append(title)
    print("\n" + "=" * 70)
    print(f">>> {title}")
    print("=" * 70)
    try:
        subprocess.run([PY, "-m", *args], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] step failed ({e}) — continuing")
        return False


def main():
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                        "--format=csv"], check=False)
    else:
        print("[!] no nvidia-smi found — is this actually a CUDA box? "
              "GPU-util numbers will be n/a and vLLM will fail.")

    ok = {}
    # 1. the throughput ladder at GPU scale (fp16), + graphs
    ok["sweep"] = step("engine x rate sweep (fp16)", [
        "bench.sweep", "--engines", "naive", "static", "continuous", "paged",
        "--rates", "4", "8", "16", "32", "64", "--n", "256", "--max-tokens", "128",
        "--device", DEV])
    ok["plot"] = step("plots", ["bench.plot"])

    # 2. deterministic memory ablation (device-independent, but re-run for the record)
    ok["memory"] = step("KV fragmentation ablation", ["bench.memory_study", "--n", "128"])

    # 3. real Azure trace at FULL lengths (len-scale 1) — natural + burst
    ok["trace"] = step("Azure trace, full lengths", [
        "bench.trace_compare", "--device", DEV, "--n", "200", "--len-scale", "1"])

    # 4. the audit rows at fp16
    ok["spec"] = step("audit: speculative decoding", ["bench.spec_study", "--device", DEV])
    ok["prefix"] = step("audit: prefix caching", ["bench.prefix_study", "--device", DEV])
    ok["kvquant"] = step("audit: KV quantization", ["bench.kv_quant_study", "--device", DEV])

    # 5. LOW-NOISE paged-vs-continuous — the comparison CPU noise couldn't resolve
    ok["noise"] = step("noise-floor: continuous vs paged (5 runs)", [
        "bench.repeat", "--compare", "continuous", "paged", "--runs", "5",
        "--rate", "32", "--n", "64", "--device", DEV])

    # 6. the reference ceiling (needs vLLM)
    ok["vllm"] = step("vLLM reference ceiling", [
        "bench.vllm_ref", "--n", "200", "--rate", "16", "--out", "results/vllm.json"])

    # 7. roofline overlay: predicted vs measured (T4 presets; override for your GPU)
    ok["roofline"] = step("roofline: predicted vs measured", [
        "bench.roofline", "--mem-bandwidth-gbps", "320", "--peak-tflops", "65",
        "--measured", "results/sweep.json"])

    print("\n" + "#" * 70)
    print("GPU RUN SUMMARY")
    for k, v in ok.items():
        print(f"  {'ok ' if v else 'FAIL'}  {k}")
    print("#" * 70)
    print("graphs + JSON in results/. Fill the writeup blanks from:")
    print("  results/sweep.json (ladder), results/vllm.json (ceiling),")
    print("  results/trace.json (real traffic), results/spec|prefix|kv_quant.json (audit)")


if __name__ == "__main__":
    main()
