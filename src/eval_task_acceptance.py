"""
Task generalization eval: SVC draft acceptance on real benchmarks.

Tests whether the bit-exact 100% acceptance from Phase 2A holds on:
  - GSM8K: math reasoning (numerical tokens, multi-step)
  - HumanEval: code completion (structured, predictable)
  - LongBench (placeholder): long-context

For each task:
  1. Sample N problems
  2. For each problem, run prefix prompt → 64 tokens of generation
  3. At each generation step, compare draft (with quantized cache) vs verify (full cache)
  4. Report per-task acceptance rate at γ ∈ {1, 2, 4, 8}

This is a more realistic test than Phase 2A — instead of greedy on wikitext,
we use real task prompts that exercise different parts of the distribution.
"""

import os
import sys
import json
import time
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn.functional as F
import numpy as np

from svc_kv_cache import DPCMCodec
from phase2a_experiment import (
    svc_quantize_cache, extract_kv_tensors, build_cache,
)


@dataclass
class EvalConfig:
    model_name: str = "Qwen/Qwen2.5-7B"
    task: str = "gsm8k"
    num_problems: int = 30
    gen_steps: int = 64               # tokens generated per problem
    max_prefix: int = 1024
    draft_lengths: tuple = (1, 2, 4, 8)
    svc_bits: int = 3
    svc_chunk_size: int = 64
    output_dir: str = "./eval_results"


cfg = EvalConfig()
cfg.model_name = os.environ.get("SVC_MODEL", cfg.model_name)
cfg.task = os.environ.get("SVC_TASK", cfg.task)
cfg.num_problems = int(os.environ.get("SVC_NUM_PROBLEMS", cfg.num_problems))
cfg.output_dir = os.environ.get("SVC_OUTDIR", cfg.output_dir)
os.makedirs(cfg.output_dir, exist_ok=True)


# ============================================================
# Task data loading
# ============================================================

def load_gsm8k(num_problems):
    """Load GSM8K problems. Falls back to a hardcoded set if dataset unavailable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        return [{"prompt": ex["question"]} for ex in ds.select(range(num_problems))]
    except Exception as e:
        print(f"  [load_dataset failed: {e}; using hardcoded fallback]")
        return [
            {"prompt": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"},
            {"prompt": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"},
            {"prompt": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"},
            {"prompt": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?"},
            {"prompt": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?"},
        ] * (num_problems // 5 + 1)


def load_humaneval(num_problems):
    try:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test")
        return [{"prompt": ex["prompt"]} for ex in ds.select(range(num_problems))]
    except Exception as e:
        print(f"  [load_dataset failed: {e}; using hardcoded fallback]")
        return [
            {"prompt": "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n    "},
            {"prompt": "def is_prime(n: int) -> bool:\n    \"\"\"Return True if n is prime.\"\"\"\n    "},
            {"prompt": "def reverse_string(s: str) -> str:\n    \"\"\"Return the reverse of s.\"\"\"\n    "},
            {"prompt": "def gcd(a: int, b: int) -> int:\n    \"\"\"Return the greatest common divisor.\"\"\"\n    "},
            {"prompt": "def quicksort(arr: list) -> list:\n    \"\"\"Sort arr in ascending order.\"\"\"\n    "},
        ] * (num_problems // 5 + 1)


# ============================================================
# Per-step draft-verify simulation
# ============================================================

@torch.no_grad()
def measure_acceptance_one_problem(model, tokenizer, prompt, cfg, device):
    """
    For one prompt, generate gen_steps tokens, comparing draft (quant cache)
    vs verify (full cache) at each step.

    Returns dict: gamma -> list of (accepted_count, gamma) per step
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=cfg.max_prefix).to(device)
    input_ids = inputs.input_ids

    # Prefill (full precision)
    out = model(input_ids, use_cache=True)
    kv_full_t = extract_kv_tensors(out.past_key_values)
    kv_quant_t = svc_quantize_cache(kv_full_t, cfg.svc_bits, cfg.svc_chunk_size)

    # Reference greedy generation (gen_steps tokens) using full cache
    ref_tokens = []
    ref_cache = build_cache(kv_full_t)
    cur = out.logits[:, -1:, :].argmax(dim=-1)
    for _ in range(cfg.gen_steps):
        ref_tokens.append(cur[0, 0].item())
        v_out = model(cur, past_key_values=ref_cache, use_cache=True)
        ref_cache = v_out.past_key_values
        cur = v_out.logits[:, -1:, :].argmax(dim=-1)

    # For each γ, sweep starting positions and measure how many tokens
    # the quant-cache draft can match before diverging
    results = {}
    for gamma in cfg.draft_lengths:
        accepts = []
        # Walk through the reference sequence; at each step build a quant cache
        # extended with ref_tokens[:t], then draft γ tokens and check matches
        # against ref_tokens[t:t+γ]
        max_starts = cfg.gen_steps - gamma
        sample_starts = list(range(0, max_starts, max(1, max_starts // 8)))[:8]

        for start in sample_starts:
            # Build a draft cache: prefill quant cache + extend with ref_tokens[:start] using FULL model
            # (extending with full to keep alignment with ref; only the prefix is quantized)
            draft_cache = build_cache(kv_quant_t)
            if start > 0:
                # Feed ref_tokens[:start] one by one into the draft cache
                ext_input = torch.tensor([ref_tokens[:start]], device=device)
                d_out0 = model(ext_input, past_key_values=draft_cache, use_cache=True)
                draft_cache = d_out0.past_key_values

            # Now draft γ tokens greedily
            draft_seq = []
            cur_d = torch.tensor([[ref_tokens[start - 1] if start > 0
                                   else ref_tokens[0]]], device=device)
            # Actually we should start from the last accepted ref token
            # Cleaner: use ref_tokens[start] as the seed and generate γ tokens
            cur_d = torch.tensor([[ref_tokens[start]]], device=device)
            d_out = model(cur_d, past_key_values=draft_cache, use_cache=True)
            draft_cache = d_out.past_key_values
            cur_d = d_out.logits[:, -1:, :].argmax(dim=-1)
            for _ in range(gamma):
                draft_seq.append(cur_d[0, 0].item())
                d_out = model(cur_d, past_key_values=draft_cache, use_cache=True)
                draft_cache = d_out.past_key_values
                cur_d = d_out.logits[:, -1:, :].argmax(dim=-1)

            # Compare to ref_tokens[start+1 : start+1+γ]
            target = ref_tokens[start + 1: start + 1 + gamma]
            accepted = 0
            for j in range(min(gamma, len(target))):
                if draft_seq[j] == target[j]:
                    accepted += 1
                else:
                    break
            accepts.append(accepted)

        results[gamma] = accepts

    return results


# ============================================================
# Main eval driver
# ============================================================

def run_eval():
    print(f"=== Task generalization eval ===")
    print(f"Model: {cfg.model_name}")
    print(f"Task:  {cfg.task}")
    print(f"Problems: {cfg.num_problems}, gen_steps: {cfg.gen_steps}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, dtype=torch.float16,
    ).cuda()
    model.eval()
    device = "cuda"
    print(f"  Loaded. Layers: {model.config.num_hidden_layers}, "
          f"KV heads: {getattr(model.config, 'num_key_value_heads', '?')}")

    # Load task data
    if cfg.task == "gsm8k":
        problems = load_gsm8k(cfg.num_problems)
    elif cfg.task == "humaneval":
        problems = load_humaneval(cfg.num_problems)
    else:
        raise ValueError(f"Unknown task: {cfg.task}")
    print(f"  Loaded {len(problems)} problems")

    # Aggregate
    all_results = {g: [] for g in cfg.draft_lengths}
    t0 = time.time()
    for i, prob in enumerate(problems):
        try:
            r = measure_acceptance_one_problem(model, tok, prob["prompt"], cfg, device)
            for g, accepts in r.items():
                all_results[g].extend(accepts)
            if (i + 1) % 5 == 0 or i == len(problems) - 1:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(problems) - i - 1)
                print(f"  [{i+1}/{len(problems)}] elapsed={elapsed:.0f}s eta={eta:.0f}s")
        except Exception as e:
            print(f"  Problem {i} failed: {e}")
            continue
        torch.cuda.empty_cache()

    # Summary
    print(f"\n=== Results: {cfg.task} on {cfg.model_name} ===")
    summary = {}
    for g in cfg.draft_lengths:
        accepts = all_results[g]
        if not accepts:
            continue
        n = len(accepts)
        mean_acc = float(np.mean(accepts))
        per_tok = mean_acc / g
        full_rate = float(np.mean([1 if a == g else 0 for a in accepts]))
        # Distribution
        from collections import Counter
        dist = Counter(accepts)
        summary[g] = {
            'samples': n,
            'mean_accepted': mean_acc,
            'per_token_accept': per_tok,
            'full_match_rate': full_rate,
            'distribution': dict(dist),
        }
        print(f"  γ={g:2d}: n={n}, mean={mean_acc:.2f}/{g}, "
              f"per-token={per_tok:.1%}, full={full_rate:.1%}")
        print(f"        distribution: {dict(sorted(dist.items()))}")

    # Save
    with open(f"{cfg.output_dir}/{cfg.task}_results.json", "w") as f:
        out_data = {
            'config': {
                'model': cfg.model_name, 'task': cfg.task,
                'num_problems': cfg.num_problems, 'gen_steps': cfg.gen_steps,
                'svc_bits': cfg.svc_bits, 'svc_chunk_size': cfg.svc_chunk_size,
            },
            'summary': {str(k): v for k, v in summary.items()},
        }
        json.dump(out_data, f, indent=2)
    print(f"\nSaved to {cfg.output_dir}/{cfg.task}_results.json")


if __name__ == "__main__":
    run_eval()
