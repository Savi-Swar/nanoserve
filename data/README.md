# Trace data

The real-workload benchmarks replay the **Azure LLM inference trace** (public,
from [Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset)).
It is not committed here (external dataset). Fetch it:

```bash
curl -sL -o data/azure_llm_conv.csv \
  https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv
```

19,366 real requests; columns `TIMESTAMP, ContextTokens, GeneratedTokens`.
Heavy-tailed: context p50=1,020 / p99=4,142 / max=14,050; output p50=129 /
p99=601. Nothing like uniform synthetic prompts — which is the point (see
`docs/writeup.md`, "Real traffic").

Then:

```bash
make trace          # all engines, natural + burst load -> results/trace.json
```
