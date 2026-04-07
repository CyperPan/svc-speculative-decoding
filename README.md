# SVC Speculative Decoding

> A KV cache compression scheme designed specifically for the **draft phase**
> of speculative decoding, where the wasted-work cost of low-precision
> failures is far smaller than in normal inference.

See [REPORT.md](REPORT.md) for the full results writeup.

## The problem

**Speculative decoding** accelerates LLM inference by letting a cheap draft
process propose γ candidate tokens that the full target model verifies in
parallel. The standard bottleneck is *draft memory*: each parallel branch
needs its own KV cache, and on long contexts this dominates HBM usage and
caps the tree width / batch size you can run.

The obvious fix is to **quantize the KV cache**. The catch is that existing
KV quantization work (KIVI, Atom, KVQuant, …) is designed for the *main
inference path*: every quantization error degrades final output quality
forever, so they spend a lot of complexity defending against worst-case
errors and still typically need 4 bits to stay safe.

Speculative decoding has a fundamentally different error budget. The draft
is *throwaway* — anything the verifier rejects is just wasted work, never
output. So a draft-side cache can run at much more aggressive precision as
long as:

1. it produces the *same* tokens as full precision on most positions, and
2. the rare divergences are cheap to detect and fall back from.

Regular KV quantization optimizes for absolute fidelity. SVC optimizes for
*verifier-aligned agreement* — a different objective that turns out to be
easier to satisfy, and lets us push down to 3 bits (~20% of FP16) without
hurting the verifier's output at all.

## What SVC actually is

A two-layer KV cache encoding inspired by Scalable Video Coding:

- **Base layer**: 3-bit chunked anchor-based DPCM. Used by the draft.
  Memory: ~20% of FP16. Per-token agreement with full cache: 95-100%.
- **Refinement layer**: FP16 residual on top of the base. Used only by the
  verifier. Adding it back gives bit-exact reconstruction.

The base layer alone is what speeds the draft up. The refinement layer
guarantees that the verifier sees the *exact* original cache, so final
output quality is mathematically identical to vanilla speculative decoding.
Memory savings show up in the draft, correctness comes from the verifier —
the asymmetry is the whole point.

## Why a new encoder, instead of reusing KIVI / Atom / KVQuant?

Three things broke when we tried the existing schemes:

1. **They target a different metric.** They minimize per-element MSE on the
   KV cache. We want to minimize *output token disagreement* with the
   reference, which is a much sparser signal — most positions tolerate
   large MSE, a few positions don't tolerate any.
2. **They're built around per-channel scaling and outlier handling**, both
   of which add overhead the draft phase can't afford.
3. **They have no notion of a refinement layer.** With them, the verifier
   would also have to consume quantized KVs, which shifts the final
   output distribution. We want the verifier to be byte-identical to
   non-speculative, so the layered design is essential.

The Phase 0 ablation in REPORT.md shows the empirical reasoning behind
chunked anchor-based DPCM specifically (rather than per-token, per-channel,
or cumulative DPCM): it's the simplest scheme that exploits the actual
redundancy structure of LLM KV caches without compounding error along the
sequence.

## Headline numbers

| Setting | KL vs full | γ=8 full-match | Memory |
|---|---|---|---|
| Phase 1 single-token (Qwen2.5-1.5B) | 1.7e-5 | — | 27% |
| Phase 2A self-speculation (Qwen2.5-7B) | — | **100%** | 20% |
| GSM8K real-task (Qwen2.5-7B) | — | 95.0% | 20% |
| HumanEval real-task (Qwen2.5-7B) | — | 95.0% | 20% |

The 5% of failures cluster at high-entropy positions (3× entropy, 2.6×
smaller logit margin than agreed positions), opening a path to a zero-cost
margin-based rejection detector that should push 95% → 99.5% without any
retraining. See `src/diagnose_failures.py` and the report.

## Layout

```
.
├── REPORT.md                       Full writeup of Phase 1 + 2A + real-task results
├── src/
│   ├── svc_kv_cache.py             SVC encoder (DPCMCodec + SVCKVCache + SVCSpeculativeDecoder)
│   ├── test_svc.py                 Phase 1 unit tests + quality measurement
│   ├── phase2a_experiment.py       Phase 2A: hidden-state quality + draft acceptance + wide tree
│   ├── eval_task_acceptance.py     GSM8K / HumanEval real-task acceptance eval
│   ├── eval_long_context.py        Long-prefix stress test
│   └── diagnose_failures.py        Per-position failure analysis (the most useful experiment)
├── scripts/                        SLURM sbatch wrappers for the NEU Explorer cluster
├── results/                        JSON outputs from past runs
│   ├── phase2a_results/                Qwen2.5-1.5B
│   ├── phase2a_results_qwen7b/         Qwen2.5-7B
│   ├── eval_results_qwen7b/            GSM8K + HumanEval
│   └── diag_results/                   Per-position diagnosis
├── requirements.txt
└── .gitignore
```

## Quick start

```bash
pip install -r requirements.txt

# Phase 1: codec sanity + quality on a real model
python src/test_svc.py

# Phase 2A: hidden-state quality, draft acceptance, wide-tree memory tradeoff
SVC_MODEL=Qwen/Qwen2.5-1.5B SVC_PREFIX_LEN=256 python src/phase2a_experiment.py

# Real-task evals
SVC_MODEL=Qwen/Qwen2.5-7B SVC_TASK=gsm8k     python src/eval_task_acceptance.py
SVC_MODEL=Qwen/Qwen2.5-7B SVC_TASK=humaneval python src/eval_task_acceptance.py

# Failure diagnosis: where and why does the quantized cache diverge?
SVC_MODEL=Qwen/Qwen2.5-7B python src/diagnose_failures.py
```

All scripts honor a few environment variables:
- `SVC_MODEL`: HuggingFace model id (default Qwen/Qwen2.5-1.5B)
- `SVC_PREFIX_LEN`: prefix length for prefill (default 256)
- `SVC_OUTDIR`: where to write JSON results
- `SVC_TASK`: `gsm8k` or `humaneval` (eval_task_acceptance.py only)
- `SVC_NUM_PROBLEMS`: how many problems to run

## How SVC works

For each layer's KV cache `[H, L, d]`:

1. Split the sequence into chunks of `chunk_size=64` tokens.
2. Each chunk's first token is kept as an FP16 anchor.
3. The other 63 tokens are stored as `(token - anchor)` residuals quantized
   uniformly to 3 bits per element.
4. Decoding is `anchor + dequant(residual)` — no cumsum, errors stay
   independent across positions, so quality does not degrade with sequence
   length.

This is the *base layer* of an SVC (Scalable Video Coding) -inspired design.
A second *refinement layer* stores the FP16 residual `(original - base)` and
gives bit-exact reconstruction when needed for verification.

The key insight from Phase 0 (in REPORT.md): KV-cache attention heads in
production LLMs are nearly orthogonal, so cross-head MDC won't work — but
the *sequence dimension* is highly redundant (50% variance captured by the
top 12-34 singular values out of 1024), and that's exactly what chunked
DPCM exploits.

## Cluster scripts

`scripts/run_*.sh` are SLURM sbatch wrappers used to run the experiments on
the NEU Explorer HPC cluster (V100-PCIe and A100 nodes). They are kept here
as concrete examples of how each experiment is invoked. To run on a
different cluster, copy one and edit the `#SBATCH` directives and the
`conda activate` line.

## Status

This is research code, not a library. The encoder and the analysis scripts
are stable; the higher-level `SVCSpeculativeDecoder` class in
`src/svc_kv_cache.py` is a sketch — the production path forward is a fused
Triton dequant+attention kernel and a vLLM integration, neither of which is
implemented here.
