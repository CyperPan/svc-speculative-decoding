# SVC Speculative Decoding

3-bit chunked-DPCM quantization of LLM KV caches that drops draft-phase
memory to ~20% of FP16 baseline while preserving 95-100% bit-exact agreement
with full-precision greedy decoding.

See [REPORT.md](REPORT.md) for the full results writeup.

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
