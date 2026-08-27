"""The Triton kernel vs SDPA, on CUDA. Three layers:

1. op-level: kernel output allclose to SDPA on random decode shapes (fp16)
2. masked op-level: left-padded batches through the additive mask path
3. model-level: greedy tokens through Qwen2.5-0.5B are IDENTICAL with the
   kernel registered vs the stock SDPA path, and the kernel-call counter
   proves the Triton path actually ran (no silent-fallback vacuous pass)

Skipped without CUDA; runs in the Kaggle pipeline (scripts/gpu_run.py, mode
"kernel").
"""
import pytest
import torch

cuda = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not cuda, reason="needs CUDA")

if cuda:
    from server.kernels import paged_attention_triton as pat


def sdpa_direct(q, k, v, mask_add=None, scale=None):
    """SDPA ground truth on GPU: repeat_kv, [B,H,1,D] query."""
    group = q.shape[1] // k.shape[1]
    kk = k.repeat_interleave(group, dim=1)
    vv = v.repeat_interleave(group, dim=1)
    am = None
    if mask_add is not None:
        am = mask_add[:, None, None, :].to(q.dtype)   # [B,1,1,T] additive
    out = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, None, :], kk, vv, attn_mask=am, scale=scale)
    return out[:, :, 0, :]


CASES = [
    (1, 14, 2, 17, 64),
    (4, 14, 2, 333, 64),
    (8, 14, 2, 2048, 64),    # the crossover-study shape
    (2, 16, 2, 512, 128),    # 3B-style head_dim
    (3, 8, 8, 129, 64),      # MHA, tile boundary + 1
    (2, 6, 2, 1, 64),
]


@pytest.mark.parametrize("B,H,Hkv,T,D", CASES)
def test_kernel_matches_sdpa(B, H, Hkv, T, D):
    torch.manual_seed(B * 100 + T)
    q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.float16)
    got = pat.decode_attention(q, k, v)
    want = sdpa_direct(q, k, v)
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("B,H,Hkv,T,D", CASES[:4])
def test_kernel_padding_mask(B, H, Hkv, T, D):
    torch.manual_seed(11)
    q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.float16)
    mask = torch.zeros(B, T, device="cuda")
    for b in range(B):
        mask[b, :(b * 61) % max(1, T - 1)] = torch.finfo(torch.float16).min
    got = pat.decode_attention(q, k, v, mask_add=mask)
    want = sdpa_direct(q, k, v, mask_add=mask)
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


def test_kernel_noncontiguous_inputs():
    """Strided views (what real cache tensors look like) go through unharmed."""
    torch.manual_seed(2)
    big = torch.randn(4, 14, 300, 64, device="cuda", dtype=torch.float16)
    q = big[:, :, 0, :]                       # strided [B,H,D] view
    k = torch.randn(4, 2, 600, 64, device="cuda", dtype=torch.float16)[:, :, ::2, :]
    v = torch.randn(4, 2, 600, 64, device="cuda", dtype=torch.float16)[:, :, ::2, :]
    got = pat.decode_attention(q, k, v)
    want = sdpa_direct(q.contiguous(), k.contiguous(), v.contiguous())
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------
# model-level: the oracle
# --------------------------------------------------------------------------
PROMPTS = [
    "The capital of France is",
    "In a distant galaxy, a small robot",
    "def fibonacci(n):",
    "The three laws of thermodynamics state",
]
N_TOKENS = 48


def greedy(runner, prompt, n):
    from server.model import sample
    from server.request import SamplingParams
    sp = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
    ids = runner.encode(prompt)
    logits, kv, cur = runner.prefill(ids)
    tok = sample(logits, sp)
    out = [tok]
    for _ in range(n - 1):
        logits, kv, cur = runner.decode(tok, kv, cur)
        tok = sample(logits, sp)
        out.append(tok)
    return out


@pytest.fixture(scope="module")
def runner():
    from server.model import ModelRunner
    return ModelRunner("Qwen/Qwen2.5-0.5B", device="cuda")


def test_model_greedy_tokens_identical(runner):
    baseline = [greedy(runner, p, N_TOKENS) for p in PROMPTS]

    calls_before = pat.KERNEL_CALLS
    prev = pat.use_triton_attention(runner.model)
    try:
        with_kernel = [greedy(runner, p, N_TOKENS) for p in PROMPTS]
    finally:
        pat.restore_attention(runner.model, prev)

    assert pat.KERNEL_CALLS > calls_before, (
        "Triton path never ran; the wrapper silently fell back to SDPA")
    for p, a, b in zip(PROMPTS, baseline, with_kernel):
        assert a == b, f"token divergence on prompt {p!r}:\n sdpa   {a}\n triton {b}"
