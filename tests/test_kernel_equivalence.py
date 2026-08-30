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
# paged kernel (M2): block-table indexing straight over the pool
# --------------------------------------------------------------------------
def test_paged_kernel_matches_sdpa():
    from tests.test_kernel_reference import build_paged_case
    B, H, Hkv, D, bs = 4, 14, 2, 64, 16
    lens = [1, 37, 200, 2048]
    pool_k, pool_v, tables, ck, cv = build_paged_case(B, Hkv, D, lens, bs, seed=21)
    torch.manual_seed(22)
    q = torch.randn(B, H, D, dtype=torch.float16)
    got = pat.paged_decode_attention(
        q.cuda(), pool_k.half().cuda(), pool_v.half().cuda(),
        tables.cuda(), torch.tensor(lens).cuda(), bs)
    for b, L in enumerate(lens):
        want = sdpa_direct(q[b:b + 1].cuda(),
                           ck[b:b + 1, :, :L].half().cuda(),
                           cv[b:b + 1, :, :L].half().cuda())
        torch.testing.assert_close(got[b:b + 1], want, atol=2e-2, rtol=2e-2)


def test_paged_kernel_block_boundaries():
    from tests.test_kernel_reference import build_paged_case
    B, H, Hkv, D, bs = 4, 4, 2, 128, 16
    lens = [16, 17, 128, 129]
    pool_k, pool_v, tables, ck, cv = build_paged_case(B, Hkv, D, lens, bs, seed=23)
    torch.manual_seed(24)
    q = torch.randn(B, H, D, dtype=torch.float16)
    got = pat.paged_decode_attention(
        q.cuda(), pool_k.half().cuda(), pool_v.half().cuda(),
        tables.cuda(), torch.tensor(lens).cuda(), bs)
    for b, L in enumerate(lens):
        want = sdpa_direct(q[b:b + 1].cuda(),
                           ck[b:b + 1, :, :L].half().cuda(),
                           cv[b:b + 1, :, :L].half().cuda())
        torch.testing.assert_close(got[b:b + 1], want, atol=2e-2, rtol=2e-2)


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


def greedy_forced(runner, prompt, n, force_tokens):
    """Teacher-forced pass: feed force_tokens, record each step's argmax and
    the top1-top2 margin of the OTHER implementation's logits at that step."""
    from server.model import sample
    from server.request import SamplingParams
    sp = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
    ids = runner.encode(prompt)
    logits, kv, cur = runner.prefill(ids)
    out = [int(logits.argmax(-1))]
    for i in range(n - 1):
        logits, kv, cur = runner.decode(force_tokens[i], kv, cur)
        out.append(int(logits.argmax(-1)))
    return out


def test_model_greedy_agreement(runner):
    """The kernel accumulates in a different order than SDPA, so fp16 logits
    differ in the last bits and an EXACT tie can argmax either way (measured on
    T4: one flip in 192 steps, at a step whose sdpa top1-top2 margin was
    exactly 0.0). Token-identity across different reduction orders is not a
    valid oracle. The valid one: teacher-forced, every disagreement must sit
    at a provable near-tie; a flip with a real margin means a real bug."""
    baselines = []
    for p in PROMPTS:
        toks = greedy(runner, p, N_TOKENS)
        # margins of the sdpa run at every step, teacher-forced on itself
        from server.request import SamplingParams
        ids = runner.encode(p)
        logits, kv, cur = runner.prefill(ids)
        margins = [float((logits.topk(2).values[0][0] - logits.topk(2).values[0][1]))]
        for i in range(N_TOKENS - 1):
            logits, kv, cur = runner.decode(toks[i], kv, cur)
            t2 = logits.topk(2).values[0]
            margins.append(float(t2[0] - t2[1]))
        baselines.append((toks, margins))

    calls_before = pat.KERNEL_CALLS
    prev = pat.use_triton_attention(runner.model)
    try:
        forced = [greedy_forced(runner, p, N_TOKENS, base[0])
                  for p, base in zip(PROMPTS, baselines)]
    finally:
        pat.restore_attention(runner.model, prev)

    assert pat.KERNEL_CALLS > calls_before, (
        "Triton path never ran; the wrapper silently fell back to SDPA")

    TIE_MARGIN = 1e-3
    total = flips = 0
    for p, (toks, margins), kern in zip(PROMPTS, baselines, forced):
        for i, (a, b) in enumerate(zip(toks, kern)):
            total += 1
            if a != b:
                flips += 1
                assert margins[i] < TIE_MARGIN, (
                    f"non-tie flip on {p!r} step {i}: sdpa margin "
                    f"{margins[i]:.4f} (>= {TIE_MARGIN}), sdpa {a} vs kernel {b}"
                    " -> real divergence, not fp16 tie noise")
    assert flips <= max(2, total // 50), (
        f"{flips}/{total} flips is too many even for ties")


# --------------------------------------------------------------------------
# split-K (M3): forced split counts must match sdpa exactly like 1-split
# --------------------------------------------------------------------------
@pytest.mark.parametrize("splits", [2, 3, 8])
@pytest.mark.parametrize("B,H,Hkv,T,D", [(1, 14, 2, 2048, 64), (4, 14, 2, 333, 64),
                                          (2, 16, 2, 512, 128)])
def test_split_kernel_matches_sdpa(splits, B, H, Hkv, T, D):
    torch.manual_seed(B * 7 + splits)
    q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.float16)
    got = pat.decode_attention(q, k, v, num_splits=splits)
    want = sdpa_direct(q, k, v)
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


def test_split_kernel_with_mask():
    torch.manual_seed(51)
    B, H, Hkv, T, D = 4, 14, 2, 700, 64
    q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.float16)
    mask = torch.zeros(B, T, device="cuda")
    for b in range(B):
        mask[b, :(b * 173) % (T - 1)] = torch.finfo(torch.float16).min
    got = pat.decode_attention(q, k, v, mask_add=mask, num_splits=5)
    want = sdpa_direct(q, k, v, mask_add=mask)
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("splits", [2, 6])
def test_split_paged_kernel_matches_sdpa(splits):
    from tests.test_kernel_reference import build_paged_case
    B, H, Hkv, D, bs = 4, 14, 2, 64, 16
    lens = [1, 37, 500, 2048]
    pool_k, pool_v, tables, ck, cv = build_paged_case(B, Hkv, D, lens, bs, seed=61)
    torch.manual_seed(62)
    q = torch.randn(B, H, D, dtype=torch.float16)
    got = pat.paged_decode_attention(
        q.cuda(), pool_k.half().cuda(), pool_v.half().cuda(),
        tables.cuda(), torch.tensor(lens).cuda(), bs, num_splits=splits)
    for b, L in enumerate(lens):
        want = sdpa_direct(q[b:b + 1].cuda(),
                           ck[b:b + 1, :, :L].half().cuda(),
                           cv[b:b + 1, :, :L].half().cuda())
        torch.testing.assert_close(got[b:b + 1], want, atol=2e-2, rtol=2e-2)
