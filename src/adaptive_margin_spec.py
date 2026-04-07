"""
Margin-aware adaptive speculative decoding with SVC.

Three modes compared head-to-head on the same prompts and the same RNG seed:

  1. vanilla       — fixed γ, no margin awareness, vanilla acceptance rule.
  2. margin_passive — fixed γ, but rejections are reclassified as
                      "real cache failures" (margin ≥ τ) vs "ambiguous"
                      (margin < τ). Output sequence is identical to vanilla.
                      Used to compute α_real, the underlying "true" cache
                      fidelity excluding ambiguity.
  3. margin_adaptive — same classifier, but γ is dynamically chosen by a
                       Kalman filter that tracks α_real (NOT the raw
                       acceptance rate). Real failures shrink γ; ambiguity
                       does not.

The key claim: vanilla's 95% acceptance is misleading because it lumps
"cache made a mistake" together with "the model itself was undecided".
Margin separation lets us recover the true cache fidelity (~99%) and
choose γ accordingly, raising tokens-per-verifier-call.

Outputs JSON with per-step records and an aggregate summary.
"""

import os
import sys
import json
import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from svc_kv_cache import DPCMCodec
from phase2a_experiment import (
    svc_quantize_cache, extract_kv_tensors, build_cache,
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-7B"
    task: str = "gsm8k"          # gsm8k or humaneval
    num_problems: int = 10
    gen_steps: int = 64
    gamma_max: int = 8
    margin_threshold: float = 4.0
    cost_ratio: float = 0.05     # draft cost / verify cost
    svc_bits: int = 3
    svc_chunk_size: int = 64
    output_dir: str = "./adaptive_results"


cfg = Config()
cfg.model_name = os.environ.get("SVC_MODEL", cfg.model_name)
cfg.task = os.environ.get("SVC_TASK", cfg.task)
cfg.num_problems = int(os.environ.get("SVC_NUM_PROBLEMS", cfg.num_problems))
cfg.margin_threshold = float(os.environ.get("SVC_MARGIN_TAU", cfg.margin_threshold))
cfg.output_dir = os.environ.get("SVC_OUTDIR", cfg.output_dir)
os.makedirs(cfg.output_dir, exist_ok=True)


# ============================================================
# Task data (same hardcoded fallbacks as eval_task_acceptance.py)
# ============================================================

GSM8K_PROMPTS = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
    "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
    "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
    "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?",
    "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
    "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file?",
    "John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home. He tries to get home in 4 hours but spends the first 2 hours in standstill traffic. He spends the next half-hour driving at a speed of 30mph, before being able to drive the remaining time of the 4 hours going at 80 mph. How far is he from home at the end of those 4 hours?",
    "Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?",
]

HUMANEVAL_PROMPTS = [
    "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n    ",
    "def is_prime(n: int) -> bool:\n    \"\"\"Return True if n is prime.\"\"\"\n    ",
    "def reverse_string(s: str) -> str:\n    \"\"\"Return the reverse of s.\"\"\"\n    ",
    "def gcd(a: int, b: int) -> int:\n    \"\"\"Return the greatest common divisor.\"\"\"\n    ",
    "def quicksort(arr: list) -> list:\n    \"\"\"Sort arr in ascending order.\"\"\"\n    ",
    "def binary_search(arr: list, target: int) -> int:\n    \"\"\"Return the index of target in sorted arr, or -1.\"\"\"\n    ",
    "def merge_sort(arr: list) -> list:\n    \"\"\"Sort arr using merge sort.\"\"\"\n    ",
    "def factorial(n: int) -> int:\n    \"\"\"Return n factorial.\"\"\"\n    ",
    "def is_palindrome(s: str) -> bool:\n    \"\"\"Return True if s reads the same forwards and backwards.\"\"\"\n    ",
    "def count_vowels(s: str) -> int:\n    \"\"\"Return the number of vowels in s.\"\"\"\n    ",
]


def load_prompts():
    if cfg.task == "gsm8k":
        return GSM8K_PROMPTS[:cfg.num_problems]
    if cfg.task == "humaneval":
        return HUMANEVAL_PROMPTS[:cfg.num_problems]
    raise ValueError(f"Unknown task {cfg.task}")


# ============================================================
# Adaptive gamma chooser (Kalman-style EWMA on alpha_real)
# ============================================================

class AlphaFilter:
    """Kalman-style filter on the binary 'real cache success' signal."""

    def __init__(self, alpha_init=0.9, P_init=0.05, Q=0.003):
        self.alpha = alpha_init
        self.P = P_init
        self.Q = Q

    def update(self, observation: float):
        """observation in [0, 1]: per-round real-cache success ratio."""
        alpha_pred = self.alpha
        P_pred = self.P + self.Q
        # Bernoulli-style observation noise, clamped
        R = max(alpha_pred * (1 - alpha_pred), 1e-3)
        K = P_pred / (P_pred + R)
        self.alpha = alpha_pred + K * (observation - alpha_pred)
        self.alpha = max(0.05, min(0.99, self.alpha))
        self.P = (1 - K) * P_pred
        return self.alpha


def optimal_gamma(alpha: float, gamma_max: int, c: float = 0.05) -> int:
    """Pick γ that maximizes E[accepted] / (γ·c + 1)."""
    if alpha < 0.05:
        return 1
    best_g, best_t = 1, 0.0
    for g in range(1, gamma_max + 1):
        if alpha >= 1.0 - 1e-9:
            expected = g
        else:
            expected = alpha * (1 - alpha ** g) / (1 - alpha)
        throughput = expected / (g * c + 1)
        if throughput > best_t:
            best_t = throughput
            best_g = g
    return best_g


# ============================================================
# Core: one round of draft + verify + margin classification
# ============================================================

@torch.no_grad()
def speculative_round(
    model, full_cache, quant_cache, gamma, prev_token_id, device
):
    """
    Run one round of speculative decoding.

    Returns:
        accepted_tokens: list[int]    — tokens to commit to the output
        rejection_info: dict or None  — {position, margin, was_real_failure} if rejected
        new_full_cache, new_quant_cache: extended caches reflecting accepted tokens
    """
    # ---- Draft phase: greedy with quantized cache ----
    draft_cache = quant_cache  # in-place; we'll restore on rollback
    draft_tokens = []
    cur = prev_token_id  # [1, 1]
    for _ in range(gamma):
        d_out = model(cur, past_key_values=draft_cache, use_cache=True)
        draft_cache = d_out.past_key_values
        cur = d_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        draft_tokens.append(cur[0, 0].item())

    # ---- Verify phase: feed all draft tokens at once with full cache ----
    # Standard speculative decoding feeds prev_token + draft_tokens[:-1] and
    # gets logits at γ positions corresponding to predictions for tokens[0..γ-1]
    verify_input = torch.tensor(
        [[prev_token_id[0, 0].item()] + draft_tokens[:-1]], device=device
    )
    v_out = model(verify_input, past_key_values=full_cache, use_cache=True)
    new_full_cache = v_out.past_key_values
    verify_logits = v_out.logits[0].float()  # [γ, vocab]

    # ---- Walk through draft tokens, recording margins ----
    accepted = []
    rejection = None
    for i in range(gamma):
        log = verify_logits[i]
        top2 = log.topk(2)
        verify_token = top2.indices[0].item()
        margin = (top2.values[0] - top2.values[1]).item()
        if draft_tokens[i] == verify_token:
            accepted.append(draft_tokens[i])
        else:
            # Rejection: take verifier's token
            accepted.append(verify_token)
            rejection = {
                'position_in_round': i,
                'margin': margin,
                'is_real_failure': margin >= cfg.margin_threshold,
                'draft_token': draft_tokens[i],
                'verify_token': verify_token,
            }
            break

    # ---- Truncate the full cache to match what we actually accepted ----
    # If we rejected at position i, we used i+1 verify positions (0..i),
    # but the cache has been extended by γ positions. We need to roll back.
    n_accepted_with_correction = len(accepted)  # i+1 if rejected, γ if not
    target_len = _cache_len(full_cache) + n_accepted_with_correction
    new_full_cache = _truncate_cache(new_full_cache, target_len)

    # The quant cache hasn't been touched yet (we extended a separate
    # draft_cache pointer). We need to extend the original quant_cache by
    # accepted tokens. Easiest path: re-encode the new full cache.
    return accepted, rejection, new_full_cache


def _cache_len(cache):
    """Get current sequence length of a HF cache."""
    if hasattr(cache, 'layers') and len(cache.layers) > 0:
        layer = cache.layers[0]
        if hasattr(layer, 'keys'):
            return layer.keys.shape[2]
    if hasattr(cache, 'key_cache'):
        return cache.key_cache[0].shape[2]
    return cache[0][0].shape[2]


def _truncate_cache(cache, target_len):
    """Truncate cache in place to first target_len positions."""
    if hasattr(cache, 'layers') and len(cache.layers) > 0:
        for layer in cache.layers:
            if hasattr(layer, 'keys'):
                layer.keys = layer.keys[:, :, :target_len, :]
                layer.values = layer.values[:, :, :target_len, :]
    elif hasattr(cache, 'key_cache'):
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][:, :, :target_len, :]
            cache.value_cache[i] = cache.value_cache[i][:, :, :target_len, :]
    return cache


# ============================================================
# Three modes: vanilla, margin_passive, margin_adaptive
# ============================================================

@torch.no_grad()
def run_mode(model, tok, prompts, mode, device):
    """
    mode in {'vanilla', 'margin_passive', 'margin_adaptive'}
    Returns aggregate stats.
    """
    all_records = []
    all_round_stats = []

    for pi, prompt in enumerate(prompts):
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=512).to(device)
        input_ids = inputs.input_ids

        # Prefill
        out = model(input_ids, use_cache=True)
        full_cache = out.past_key_values
        prev_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # State
        if mode == 'margin_adaptive':
            alpha_filter = AlphaFilter(alpha_init=0.9)
            gamma = cfg.gamma_max
        else:
            alpha_filter = None
            gamma = cfg.gamma_max

        produced = 0
        rounds = []
        while produced < cfg.gen_steps:
            # Re-encode quant cache from current full cache
            kv_full_t = extract_kv_tensors(full_cache)
            kv_quant_t = svc_quantize_cache(kv_full_t, cfg.svc_bits, cfg.svc_chunk_size)
            quant_cache = build_cache(kv_quant_t)

            this_gamma = min(gamma, cfg.gen_steps - produced)
            accepted, rejection, full_cache = speculative_round(
                model, full_cache, quant_cache, this_gamma, prev_token, device
            )
            produced += len(accepted)
            prev_token = torch.tensor([[accepted[-1]]], device=device)

            # Per-round bookkeeping
            n_acc = len(accepted)
            had_rejection = rejection is not None
            real_failure = rejection['is_real_failure'] if had_rejection else False
            ambiguous = had_rejection and not real_failure
            margin = rejection['margin'] if had_rejection else None

            round_rec = {
                'gamma_used': this_gamma,
                'accepted_count': n_acc,
                'all_accepted': not had_rejection,
                'rejection_position': rejection['position_in_round'] if had_rejection else None,
                'rejection_margin': margin,
                'is_real_failure': real_failure,
                'is_ambiguous': ambiguous,
            }

            # Adaptive update (only for adaptive mode)
            if mode == 'margin_adaptive':
                # Per-round "real success ratio":
                # treat ambiguous rejections as success (the cache wasn't wrong),
                # treat real failures as failure
                if not had_rejection:
                    obs = 1.0
                elif ambiguous:
                    obs = (n_acc - 1) / this_gamma + 1.0 / this_gamma  # full credit
                    # Equivalent: ambiguous rejections don't penalize alpha_real
                    obs = 1.0
                else:
                    obs = (n_acc - 1) / this_gamma  # exclude the rejected token
                    obs = max(0.0, obs)
                alpha_filter.update(obs)
                gamma = optimal_gamma(alpha_filter.alpha, cfg.gamma_max, cfg.cost_ratio)
                round_rec['alpha_real'] = alpha_filter.alpha
                round_rec['next_gamma'] = gamma

            rounds.append(round_rec)
            torch.cuda.empty_cache()

        all_round_stats.append({
            'problem': pi,
            'rounds': rounds,
            'total_produced': produced,
            'verifier_calls': len(rounds),
        })

        del out, full_cache, quant_cache
        torch.cuda.empty_cache()
        print(f"  [{mode}] problem {pi+1}/{len(prompts)}: "
              f"{len(rounds)} rounds, {produced} tokens, "
              f"tokens/call={produced/len(rounds):.2f}")

    # Aggregate
    total_tokens = sum(r['total_produced'] for r in all_round_stats)
    total_calls = sum(r['verifier_calls'] for r in all_round_stats)
    all_rounds_flat = [rr for r in all_round_stats for rr in r['rounds']]
    n_rounds = len(all_rounds_flat)
    n_full_accept = sum(1 for r in all_rounds_flat if r['all_accepted'])
    n_real_failure = sum(1 for r in all_rounds_flat if r['is_real_failure'])
    n_ambiguous = sum(1 for r in all_rounds_flat if r['is_ambiguous'])

    agg = {
        'mode': mode,
        'tokens_per_call': total_tokens / total_calls,
        'total_tokens': total_tokens,
        'total_verifier_calls': total_calls,
        'rounds': n_rounds,
        'full_accept_rounds': n_full_accept,
        'real_failure_rounds': n_real_failure,
        'ambiguous_rounds': n_ambiguous,
        'full_accept_rate': n_full_accept / n_rounds,
        'real_failure_rate': n_real_failure / n_rounds,
        'ambiguous_rate': n_ambiguous / n_rounds,
    }

    if mode == 'margin_adaptive':
        gammas = [r['gamma_used'] for r in all_rounds_flat]
        alphas = [r.get('alpha_real') for r in all_rounds_flat if r.get('alpha_real') is not None]
        agg['mean_gamma'] = float(np.mean(gammas))
        agg['mean_alpha_real'] = float(np.mean(alphas)) if alphas else None
        agg['gamma_distribution'] = {int(g): gammas.count(g) for g in sorted(set(gammas))}

    return agg, all_round_stats


# ============================================================
# Main
# ============================================================

def main():
    print(f"=== Adaptive margin-aware SVC eval ===")
    print(f"Model: {cfg.model_name}")
    print(f"Task: {cfg.task}, problems: {cfg.num_problems}, gen_steps: {cfg.gen_steps}")
    print(f"γ_max: {cfg.gamma_max}, margin τ: {cfg.margin_threshold}, c: {cfg.cost_ratio}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, dtype=torch.float16,
    ).cuda()
    model.eval()
    print(f"  Model loaded")

    prompts = load_prompts()

    results = {}
    for mode in ['vanilla', 'margin_passive', 'margin_adaptive']:
        print(f"\n--- Mode: {mode} ---")
        t0 = time.time()
        agg, details = run_mode(model, tok, prompts, mode, "cuda")
        elapsed = time.time() - t0
        results[mode] = {'aggregate': agg, 'details': details, 'elapsed_s': elapsed}
        print(f"  Done in {elapsed:.0f}s")
        print(f"  tokens/call = {agg['tokens_per_call']:.3f}")
        print(f"  full_accept = {agg['full_accept_rate']:.1%}")
        print(f"  real_failure = {agg['real_failure_rate']:.1%}")
        print(f"  ambiguous = {agg['ambiguous_rate']:.1%}")
        if 'mean_gamma' in agg:
            print(f"  mean γ = {agg['mean_gamma']:.2f}, mean α_real = {agg['mean_alpha_real']:.3f}")
            print(f"  γ distribution: {agg['gamma_distribution']}")

    # Comparison
    print("\n=== Comparison ===")
    v = results['vanilla']['aggregate']
    p = results['margin_passive']['aggregate']
    a = results['margin_adaptive']['aggregate']
    print(f"  Vanilla        : tokens/call={v['tokens_per_call']:.3f}, full={v['full_accept_rate']:.1%}")
    print(f"  Margin passive : tokens/call={p['tokens_per_call']:.3f}, "
          f"real_fail={p['real_failure_rate']:.1%}, ambig={p['ambiguous_rate']:.1%}")
    print(f"  Margin adaptive: tokens/call={a['tokens_per_call']:.3f}, "
          f"mean_γ={a['mean_gamma']:.2f}, α_real={a['mean_alpha_real']:.3f}")
    speedup = a['tokens_per_call'] / v['tokens_per_call']
    print(f"\n  Adaptive vs Vanilla speedup: {speedup:.3f}×")

    # Save
    out_path = f"{cfg.output_dir}/{cfg.task}_adaptive.json"
    save_data = {
        'config': {
            'model': cfg.model_name, 'task': cfg.task,
            'num_problems': cfg.num_problems, 'gen_steps': cfg.gen_steps,
            'gamma_max': cfg.gamma_max, 'margin_threshold': cfg.margin_threshold,
            'cost_ratio': cfg.cost_ratio,
        },
        'results': {m: results[m]['aggregate'] for m in results},
        'speedup_adaptive_vs_vanilla': speedup,
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
