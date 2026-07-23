# Running the headline numbers on a GPU

The CPU dev box proves correctness and the *relative* story; absolute fp16
throughput and the vLLM ceiling need a real **NVIDIA/CUDA** GPU. An Intel GPU
(integrated or Arc) on Windows is not a viable path — vLLM won't run and the
Torch XPU backend is immature. Use a free **Colab T4** or a cheap rented A10/L4
instead. Everything below is CUDA; no code changes.

One command does the whole run once the code is on the box:

```bash
pip install -r requirements.txt && pip install vllm
python scripts/gpu_run.py       # sweep + trace + audit + noise + vLLM + roofline
```

## Option A — Google Colab (free T4, $0)

1. colab.research.google.com → New notebook.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Get the code onto Colab (pick one):
   - **From GitHub** (after you push — see below):
     ```python
     !git clone https://github.com/<you>/nanoserve.git
     %cd nanoserve
     ```
   - **Zip upload** (no GitHub): zip the `nanoserve` folder, use the Files panel
     (📁 left sidebar) to upload, then:
     ```python
     !unzip -q nanoserve.zip && cd nanoserve
     ```
4. Install + fetch the trace + run:
   ```python
   !pip install -q -r requirements.txt && pip install -q vllm
   !curl -sL -o data/azure_llm_conv.csv https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv
   !python scripts/gpu_run.py
   ```
5. Show the graphs inline:
   ```python
   from IPython.display import Image, display
   for f in ["throughput_vs_rate", "ttft_p99_by_engine", "throughput_by_engine", "latency_throughput"]:
       display(Image(f"results/{f}.png"))
   ```

Note: a T4 is fp16-capable but modest; vLLM runs but is older-GPU slow. The
ladder, the ceiling, and the low-noise paged-vs-continuous verdict all come out
clean. For faster/bigger runs rent an A10/L4 (~$0.40–0.75/hr, RunPod/Lambda/Vast)
— same commands, just add more `--rates`/`--n`.

## Option B — push to GitHub first (recommended; it's a portfolio asset)

From the `nanoserve` folder on your machine:

```bash
git init && git add -A && git commit -m "nanoserve: from-scratch inference server + audit"
gh repo create nanoserve --public --source=. --push     # or create the repo in the web UI and:
# git remote add origin https://github.com/<you>/nanoserve.git && git push -u origin main
```

The `.gitignore` already excludes the model cache, the 700 KB Azure trace
(re-fetched by the runbook), and generated PNGs/JSON — so the push is just code
+ docs. CI (`.github/workflows/ci.yml`) runs on the first push.

## What to expect out

`results/` fills with the graphs and JSON. Then fill the `__` blanks in
`docs/writeup.md`:
- `sweep.json` → the fp16 throughput ladder + p99 TTFT
- `vllm.json` → the `% of vLLM` ceiling number
- the `repeat` verdict → paged vs continuous, now past a real (low) noise floor
- `trace.json` → the real-traffic findings at full lengths
- `roofline` overlay → predicted vs measured, and the gap = scheduler/kernel overhead
