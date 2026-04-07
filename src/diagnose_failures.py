"""
Failure diagnosis: find WHERE and WHY SVC quantized cache diverges from full cache.

For each problem in a task:
  1. Run greedy generation with both full and quant caches
  2. At every position, record:
     - Did the top-1 token agree?
     - What was the top-1 probability under each?
     - What was the entropy of the next-token distribution?
     - What was the L2 norm of the hidden state delta?

Plot/output:
  - Distribution of disagreement positions
  - Per-position entropy at disagreements vs agreements
  - Token type analysis (numeric, identifier, punctuation)
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np

from svc_kv_cache import DPCMCodec
from phase2a_experiment import (
    svc_quantize_cache, extract_kv_tensors, build_cache,
)

# Same hardcoded GSM8K problems as eval_task_acceptance.py for reproducibility
GSM8K_PROMPTS = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
    "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
    "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
]


@torch.no_grad()
def diagnose(model, tok, prompt, gen_steps, device):
    """For one prompt, generate gen_steps tokens with both caches and record per-position info."""
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    input_ids = inputs.input_ids

    # Prefill (full)
    out = model(input_ids, use_cache=True)
    kv_full_t = extract_kv_tensors(out.past_key_values)
    kv_quant_t = svc_quantize_cache(kv_full_t, 3, 64)

    # Run two parallel greedy generations:
    full_cache = build_cache(kv_full_t)
    quant_cache = build_cache(kv_quant_t)

    cur_full = out.logits[:, -1:, :].argmax(dim=-1)
    cur_quant = cur_full.clone()  # start aligned

    records = []
    for step in range(gen_steps):
        # Forward both caches with the SAME input token (the one each chose last step)
        f_out = model(cur_full, past_key_values=full_cache, use_cache=True)
        full_cache = f_out.past_key_values
        f_logits = f_out.logits[0, -1, :].float()
        f_probs = F.softmax(f_logits, dim=-1)
        f_top = f_probs.argmax().item()
        f_top_p = f_probs[f_top].item()
        f_entropy = -(f_probs * (f_probs + 1e-10).log()).sum().item()

        q_out = model(cur_quant, past_key_values=quant_cache, use_cache=True)
        quant_cache = q_out.past_key_values
        q_logits = q_out.logits[0, -1, :].float()
        q_probs = F.softmax(q_logits, dim=-1)
        q_top = q_probs.argmax().item()
        q_top_p = q_probs[q_top].item()

        # KL between distributions
        log_q = F.log_softmax(q_logits, dim=-1)
        log_f = F.log_softmax(f_logits, dim=-1)
        mask = f_probs > 1e-8
        kl = (f_probs[mask] * (log_f[mask] - log_q[mask])).sum().item()

        # Logit-level error on top-1 of full
        f_top_logit_full = f_logits[f_top].item()
        f_top_logit_quant = q_logits[f_top].item()
        # Margin: how far is full's top-1 above the next contender under quant?
        q_top2 = q_logits.topk(2).indices
        margin_quant = (q_logits[q_top2[0]] - q_logits[q_top2[1]]).item()
        margin_full = (f_logits.topk(2).values[0] - f_logits.topk(2).values[1]).item()

        records.append({
            'step': step,
            'full_top1': f_top,
            'quant_top1': q_top,
            'agree': f_top == q_top,
            'full_top1_prob': f_top_p,
            'quant_top1_prob': q_top_p,
            'full_entropy': f_entropy,
            'kl': kl,
            'full_margin': margin_full,
            'quant_margin': margin_quant,
            'token_str': tok.decode([f_top]),
        })

        # Advance both with their own choice (this is the key: divergence
        # propagates because each cache continues its own trajectory)
        cur_full = torch.tensor([[f_top]], device=device)
        cur_quant = torch.tensor([[q_top]], device=device)

    return records


def run():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = os.environ.get("SVC_MODEL", "Qwen/Qwen2.5-7B")
    print(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16,
    ).cuda()
    model.eval()

    out_dir = os.environ.get("SVC_OUTDIR", "./diag_results")
    os.makedirs(out_dir, exist_ok=True)

    all_records = []
    for i, prompt in enumerate(GSM8K_PROMPTS):
        print(f"\nProblem {i+1}: {prompt[:80]}...")
        recs = diagnose(model, tok, prompt, gen_steps=64, device="cuda")
        torch.cuda.empty_cache()

        # Find first disagreement
        first_disagree = None
        for r in recs:
            if not r['agree']:
                first_disagree = r['step']
                break

        agreed = sum(1 for r in recs if r['agree'])
        print(f"  Agreed {agreed}/64 steps. First disagreement: step {first_disagree}")

        if first_disagree is not None:
            r = recs[first_disagree]
            print(f"    @ step {first_disagree}: full={r['full_top1']!r}({r['token_str']!r}) "
                  f"quant={r['quant_top1']!r}")
            print(f"    full_top1_p={r['full_top1_prob']:.4f}, "
                  f"entropy={r['full_entropy']:.3f}, KL={r['kl']:.4f}")
            print(f"    full_margin={r['full_margin']:.3f}, quant_margin={r['quant_margin']:.3f}")

        all_records.append({'problem_idx': i, 'records': recs})

    with open(f"{out_dir}/diagnose.json", "w") as f:
        json.dump(all_records, f, indent=2, default=str)
    print(f"\nSaved to {out_dir}/diagnose.json")

    # Aggregate stats
    print("\n=== Aggregate ===")
    all_steps = [r for prob in all_records for r in prob['records']]
    agreed = [r for r in all_steps if r['agree']]
    disagreed = [r for r in all_steps if not r['agree']]
    print(f"Total: {len(all_steps)} steps, {len(agreed)} agreed, {len(disagreed)} disagreed")
    if disagreed:
        print(f"Disagreed:  mean entropy = {np.mean([r['full_entropy'] for r in disagreed]):.3f}, "
              f"mean full_margin = {np.mean([r['full_margin'] for r in disagreed]):.3f}, "
              f"mean kl = {np.mean([r['kl'] for r in disagreed]):.4f}")
    if agreed:
        print(f"Agreed:     mean entropy = {np.mean([r['full_entropy'] for r in agreed]):.3f}, "
              f"mean full_margin = {np.mean([r['full_margin'] for r in agreed]):.3f}, "
              f"mean kl = {np.mean([r['kl'] for r in agreed]):.4f}")


if __name__ == "__main__":
    run()
