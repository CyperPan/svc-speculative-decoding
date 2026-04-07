"""
Phase 2A: EAGLE + SVC Quantized KV Cache Integration
=====================================================

Goal: Measure whether SVC's 3-bit base cache can accelerate EAGLE's draft phase
without degrading draft quality.

Three sub-experiments:
  2A.1: Hidden state quality — does quantized cache produce usable features?
  2A.2: Draft tree acceptance — quantized cache vs full cache
  2A.3: Wide tree tradeoff — use memory savings for wider draft trees
"""

import os
import json
import time
import gc
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# Import the Phase 1 vectorized SVC encoder
from svc_kv_cache import DPCMCodec


# ============================================================
# 0. Configuration
# ============================================================

@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-1.5B"

    # Dataset
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"

    # SVC params
    svc_bits: int = 3
    svc_chunk_size: int = 64

    # Experiment params
    prefix_len: int = 256
    num_samples: int = 4
    draft_lengths: list = field(default_factory=lambda: [1, 2, 4, 8])
    tree_widths: list = field(default_factory=lambda: [1, 4, 8, 16])

    output_dir: str = "./phase2a_results"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


cfg = Config()

# Allow override via environment variables for cluster runs
cfg.model_name = os.environ.get("SVC_MODEL", cfg.model_name)
cfg.prefix_len = int(os.environ.get("SVC_PREFIX_LEN", cfg.prefix_len))
cfg.output_dir = os.environ.get("SVC_OUTDIR", cfg.output_dir)
os.makedirs(cfg.output_dir, exist_ok=True)


# ============================================================
# 1. SVC Cache Quantization (vectorized, from Phase 1)
# ============================================================

def svc_quantize_kv_layer(k, v, codec):
    """
    Quantize one layer's K and V via Phase 1 codec.

    Args:
      k, v: [batch=1, num_kv_heads, seq_len, head_dim]
      codec: DPCMCodec instance

    Returns: dequantized (k_hat, v_hat) — same shape and dtype
    """
    # Strip batch, encode, decode, restore batch
    k_in = k[0].float()  # [H, L, d]
    v_in = v[0].float()

    k_codes, k_scales, k_offsets, k_anchors = codec.encode(k_in)
    v_codes, v_scales, v_offsets, v_anchors = codec.encode(v_in)

    k_hat = codec.decode(k_codes, k_scales, k_offsets, k_anchors)
    v_hat = codec.decode(v_codes, v_scales, v_offsets, v_anchors)

    return k_hat.unsqueeze(0).to(k.dtype), v_hat.unsqueeze(0).to(v.dtype)


def svc_quantize_cache(kv_list, bits=3, chunk_size=64):
    """Quantize a full KV cache (list of (k, v) tuples)."""
    codec = DPCMCodec(bits=bits, chunk_size=chunk_size)
    return [svc_quantize_kv_layer(k, v, codec) for k, v in kv_list]


# ============================================================
# 2. Cache utilities (transformers 5.x compatible)
# ============================================================

def get_num_layers(past_kv):
    if hasattr(past_kv, 'layers') and len(past_kv.layers) > 0:
        return len(past_kv.layers)
    if hasattr(past_kv, 'key_cache') and past_kv.key_cache is not None:
        try:
            return len(past_kv.key_cache)
        except Exception:
            pass
    return len(past_kv)


def _get_layer_kv(past_kv, layer_idx):
    if hasattr(past_kv, 'layers') and len(past_kv.layers) > 0:
        layer = past_kv.layers[layer_idx]
        if hasattr(layer, 'keys'):
            return layer.keys, layer.values
    if hasattr(past_kv, 'key_cache') and past_kv.key_cache is not None:
        return past_kv.key_cache[layer_idx], past_kv.value_cache[layer_idx]
    return past_kv[layer_idx][0], past_kv[layer_idx][1]


def extract_kv_tensors(past_kv):
    out = []
    for i in range(get_num_layers(past_kv)):
        k, v = _get_layer_kv(past_kv, i)
        out.append((k.detach().clone(), v.detach().clone()))
    return out


def build_cache(kv_list):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for li, (k, v) in enumerate(kv_list):
        cache.update(k, v, layer_idx=li)
    return cache


# ============================================================
# 3. Hidden State Capture
# ============================================================

class HiddenStateCapture:
    def __init__(self, model, layer_idx=-2):
        self.hidden_states = None
        self.hook = None
        self.model = model
        self.layer_idx = layer_idx

    def _get_layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        raise ValueError("Cannot find transformer layers")

    def __enter__(self):
        layers = self._get_layers()
        idx = self.layer_idx if self.layer_idx >= 0 else len(layers) + self.layer_idx
        layer = layers[idx]

        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.hidden_states = h.detach()

        self.hook = layer.register_forward_hook(hook_fn)
        return self

    def __exit__(self, *args):
        if self.hook is not None:
            self.hook.remove()


# ============================================================
# 4. Experiment 2A.1: Hidden State Quality
# ============================================================

@torch.no_grad()
def exp2a1_hidden_state_quality(model, samples, cfg):
    print("\n" + "=" * 60)
    print("Exp 2A.1: Hidden State Quality (Full vs Quantized Cache)")
    print("=" * 60)

    results = {'per_sample': []}

    with HiddenStateCapture(model, layer_idx=-2) as cap:
        for si, sample in enumerate(samples):
            prefix = sample[:, :cfg.prefix_len].to(cfg.device)

            # Reference: full forward → grab hidden state at last position
            out_full = model(prefix, use_cache=True)
            hidden_full_last = cap.hidden_states[0, -1, :].float().cpu()
            logits_full = out_full.logits[0, -1, :].float().cpu()

            # Quantize cache; truncate by 1 so we can re-run last token
            kv_full = extract_kv_tensors(out_full.past_key_values)
            kv_quant = svc_quantize_cache(kv_full, cfg.svc_bits, cfg.svc_chunk_size)
            kv_quant_trunc = [(k[:, :, :-1, :], v[:, :, :-1, :]) for k, v in kv_quant]
            trunc_cache = build_cache(kv_quant_trunc)

            last_token = prefix[:, -1:]
            out_quant = model(last_token, past_key_values=trunc_cache, use_cache=True)
            hidden_quant_last = cap.hidden_states[0, -1, :].float().cpu()
            logits_quant = out_quant.logits[0, -1, :].float().cpu()

            # Metrics
            cos_sim = F.cosine_similarity(
                hidden_full_last.unsqueeze(0), hidden_quant_last.unsqueeze(0)
            ).item()
            l2_rel = ((hidden_full_last - hidden_quant_last).norm() /
                      (hidden_full_last.norm() + 1e-10)).item()

            log_p_full = F.log_softmax(logits_full, dim=-1)
            log_p_quant = F.log_softmax(logits_quant, dim=-1)
            p_full = log_p_full.exp()
            mask = p_full > 1e-8
            kl = (p_full[mask] * (log_p_full[mask] - log_p_quant[mask])).sum().item()

            top5_full = logits_full.topk(5).indices.tolist()
            top5_quant = logits_quant.topk(5).indices.tolist()
            top1_match = top5_full[0] == top5_quant[0]
            top5_overlap = len(set(top5_full) & set(top5_quant)) / 5

            r = {
                'cos_sim': cos_sim,
                'l2_relative_error': l2_rel,
                'logit_kl': kl,
                'top1_match': top1_match,
                'top5_overlap': top5_overlap,
            }
            results['per_sample'].append(r)
            print(f"  Sample {si+1}: cos={cos_sim:.6f}, L2_rel={l2_rel:.6f}, "
                  f"KL={kl:.6f}, Top1={'✓' if top1_match else '✗'}, "
                  f"Top5={top5_overlap:.0%}")

            del out_full, out_quant, kv_full, kv_quant, trunc_cache
            torch.cuda.empty_cache()

    s = results['per_sample']
    results['summary'] = {
        'cos_sim_mean': float(np.mean([r['cos_sim'] for r in s])),
        'l2_rel_mean': float(np.mean([r['l2_relative_error'] for r in s])),
        'kl_mean': float(np.mean([r['logit_kl'] for r in s])),
        'top1_rate': float(np.mean([r['top1_match'] for r in s])),
        'top5_mean': float(np.mean([r['top5_overlap'] for r in s])),
    }
    summ = results['summary']
    print(f"\n  Summary: cos={summ['cos_sim_mean']:.6f}, "
          f"L2_rel={summ['l2_rel_mean']:.6f}, KL={summ['kl_mean']:.6f}, "
          f"Top1={summ['top1_rate']:.0%}, Top5={summ['top5_mean']:.0%}")
    return results


# ============================================================
# 5. Experiment 2A.2: Draft Acceptance
# ============================================================

@torch.no_grad()
def exp2a2_draft_acceptance(model, samples, cfg):
    print("\n" + "=" * 60)
    print("Exp 2A.2: Simulated Draft-Verify Acceptance Rate")
    print("=" * 60)

    results = {}

    for gamma in cfg.draft_lengths:
        per_sample = []
        for si, sample in enumerate(samples):
            prefix = sample[:, :cfg.prefix_len].to(cfg.device)

            out = model(prefix, use_cache=True)
            kv_full = extract_kv_tensors(out.past_key_values)
            kv_quant = svc_quantize_cache(kv_full, cfg.svc_bits, cfg.svc_chunk_size)
            first_logits = out.logits[:, -1, :]

            # Draft with quantized cache (greedy)
            draft_cache = build_cache(kv_quant)
            next_id = first_logits.argmax(dim=-1, keepdim=True)  # [1,1]
            draft_tokens = []
            for _ in range(gamma):
                d_out = model(next_id, past_key_values=draft_cache, use_cache=True)
                draft_cache = d_out.past_key_values
                next_id = d_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                draft_tokens.append(next_id[0, 0].item())

            # Verify with full-precision cache (greedy)
            verify_cache = build_cache(kv_full)
            verify_id = first_logits.argmax(dim=-1, keepdim=True)
            ref_tokens = []
            for _ in range(gamma):
                v_out = model(verify_id, past_key_values=verify_cache, use_cache=True)
                verify_cache = v_out.past_key_values
                verify_id = v_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ref_tokens.append(verify_id[0, 0].item())

            # Count accepted prefix
            accepted = 0
            for t in range(gamma):
                if draft_tokens[t] == ref_tokens[t]:
                    accepted += 1
                else:
                    break

            per_sample.append({'accepted': accepted, 'gamma': gamma})
            del out, draft_cache, verify_cache, kv_full, kv_quant
            torch.cuda.empty_cache()

        results[gamma] = {
            'mean_accepted': float(np.mean([r['accepted'] for r in per_sample])),
            'mean_acceptance_rate': float(np.mean([r['accepted'] / gamma for r in per_sample])),
            'full_accept_rate': float(np.mean([1 if r['accepted'] == gamma else 0
                                                for r in per_sample])),
        }
        r = results[gamma]
        print(f"  γ={gamma:2d}: accepted={r['mean_accepted']:.2f}/{gamma}, "
              f"per-token={r['mean_acceptance_rate']:.1%}, "
              f"full={r['full_accept_rate']:.0%}")
    return results


# ============================================================
# 6. Experiment 2A.3: Wide Tree
# ============================================================

@torch.no_grad()
def exp2a3_wide_tree(model, samples, cfg):
    print("\n" + "=" * 60)
    print("Exp 2A.3: Wide Tree Tradeoff (γ=4, T=0.8)")
    print("=" * 60)

    gamma = 4
    temperature = 0.8
    results = {}

    for width in cfg.tree_widths:
        per_sample = []
        for si, sample in enumerate(samples[:3]):  # Fewer samples for speed
            prefix = sample[:, :cfg.prefix_len].to(cfg.device)
            out = model(prefix, use_cache=True)
            kv_full = extract_kv_tensors(out.past_key_values)
            kv_quant = svc_quantize_cache(kv_full, cfg.svc_bits, cfg.svc_chunk_size)
            first_logits = out.logits[:, -1, :].float()

            # W draft branches with temperature sampling
            branches = []
            for _ in range(width):
                draft_cache = build_cache(kv_quant)
                probs = F.softmax(first_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs[0], 1).unsqueeze(0)
                tokens = [next_id[0, 0].item()]
                for _ in range(gamma - 1):
                    d_out = model(next_id, past_key_values=draft_cache, use_cache=True)
                    draft_cache = d_out.past_key_values
                    p = F.softmax(d_out.logits[:, -1, :].float() / temperature, dim=-1)
                    next_id = torch.multinomial(p[0], 1).unsqueeze(0)
                    tokens.append(next_id[0, 0].item())
                branches.append(tokens)
                del draft_cache

            # Greedy reference with full cache.
            # ref[0] = argmax(first_logits)  (the same "first new token" the draft sampled).
            # ref[1..γ-1] = greedy continuation by feeding ref[i-1] to the model.
            verify_cache = build_cache(kv_full)
            ref = [first_logits.argmax(dim=-1).item()]
            verify_id = torch.tensor([[ref[0]]], device=cfg.device)
            for _ in range(gamma - 1):
                v_out = model(verify_id, past_key_values=verify_cache, use_cache=True)
                verify_cache = v_out.past_key_values
                verify_id = v_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ref.append(verify_id[0, 0].item())

            best_match = 0
            any_full = False
            for branch in branches:
                m = 0
                for j in range(gamma):
                    if branch[j] == ref[j]:
                        m += 1
                    else:
                        break
                best_match = max(best_match, m)
                if m == gamma:
                    any_full = True

            per_sample.append({
                'best_match': best_match,
                'any_full_match': any_full,
                'unique_first_tokens': len(set(b[0] for b in branches)),
            })
            del out, verify_cache, kv_full, kv_quant
            torch.cuda.empty_cache()

        results[width] = {
            'mean_best_match': float(np.mean([r['best_match'] for r in per_sample])),
            'any_full_match_rate': float(np.mean([r['any_full_match'] for r in per_sample])),
            'mean_unique_first': float(np.mean([r['unique_first_tokens'] for r in per_sample])),
        }
        r = results[width]
        print(f"  W={width:3d}: best={r['mean_best_match']:.2f}/{gamma}, "
              f"P(any_full)={r['any_full_match_rate']:.0%}, "
              f"unique_first={r['mean_unique_first']:.1f}")
    return results


# ============================================================
# 7. Memory Budget
# ============================================================

def analyze_memory_budget(model, cfg):
    print("\n" + "=" * 60)
    print("Memory Budget Analysis")
    print("=" * 60)

    n_layers = model.config.num_hidden_layers
    n_kv = getattr(model.config, 'num_key_value_heads',
                   model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    L = cfg.prefix_len

    full_mb = n_layers * n_kv * L * head_dim * 2 * 2 / 1e6  # K+V FP16
    n_chunks = (L + cfg.svc_chunk_size - 1) // cfg.svc_chunk_size
    quant_mb = n_layers * n_kv * head_dim * 2 * (
        n_chunks * 2 + (L - n_chunks) * cfg.svc_bits / 8
    ) / 1e6
    branch_mb = n_layers * n_kv * max(cfg.draft_lengths) * head_dim * 2 * 2 / 1e6

    savings = full_mb - quant_mb
    extra_branches = savings / branch_mb if branch_mb > 0 else 0

    print(f"  Model: {cfg.model_name}, prefix_len={L}")
    print(f"  Full cache:      {full_mb:.1f} MB")
    print(f"  Quant cache:     {quant_mb:.1f} MB ({100*quant_mb/full_mb:.0f}%)")
    print(f"  Savings:         {savings:.1f} MB")
    print(f"  Per-branch cost: {branch_mb:.2f} MB (γ={max(cfg.draft_lengths)})")
    print(f"  Extra branches:  {extra_branches:.0f}")

    return {
        'full_cache_mb': full_mb, 'quant_cache_mb': quant_mb,
        'savings_mb': savings, 'branch_cost_mb': branch_mb,
        'extra_branches': extra_branches,
    }


# ============================================================
# 8. Main
# ============================================================

def save_json(data, name):
    def conv(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [conv(v) for v in o]
        return o
    with open(f"{cfg.output_dir}/{name}.json", 'w') as f:
        json.dump(conv(data), f, indent=2)


SAMPLE_TEXTS = [
    """The history of artificial intelligence began in antiquity, with myths, stories
and rumors of artificial beings endowed with intelligence or consciousness by master
craftsmen. The seeds of modern AI were planted by classical philosophers who attempted
to describe the process of human thinking as the mechanical manipulation of symbols.
This work culminated in the invention of the programmable digital computer in the 1940s,
a machine based on the abstract essence of mathematical reasoning. This device and the
ideas behind it inspired a handful of scientists to begin seriously discussing the
possibility of building an electronic brain. The field of AI research was founded at
a workshop held on the campus of Dartmouth College during the summer of 1956. Those
who attended would become the leaders of AI research for decades. Many of them
predicted that a machine as intelligent as a human being would exist in no more than
a generation, and they were given millions of dollars to make this vision come true.""",

    """In computer science, a transformer is a deep learning architecture that relies
on the parallel multi-head attention mechanism. The modern transformer was proposed in
the 2017 paper titled Attention Is All You Need. Text is converted to numerical
representations called tokens, and each token is converted into a vector via lookup
from a word embedding table. At each layer, each token is then contextualized within
the scope of the context window with other tokens via a parallel multi-head attention
mechanism, allowing the signal for key tokens to be amplified and less important tokens
to be diminished. Transformers have the advantage of having no recurrent units, requiring
less training time than previous recurrent neural architectures, such as long short-term
memory. Later variations have been widely adopted for training large language models on
large text-based datasets. The modern version of the transformer was proposed by Google.""",

    """A large language model is a language model notable for its ability to achieve
general-purpose language generation and other natural language processing tasks such as
classification. Based on language models, LLMs acquire these abilities by learning
statistical relationships from text documents during a computationally intensive
self-supervised and semi-supervised training process. LLMs can be used for text generation,
a form of generative AI, by taking an input text and repeatedly predicting the next token
or word. LLMs are artificial neural networks. The largest and most capable, as of August
2024, are built with a decoder-only transformer-based architecture, which enables
efficient processing and generation of large-scale text data. Modern models can be
fine-tuned for specific tasks or guided by prompt engineering. These models acquire
predictive power regarding syntax, semantics, and ontologies inherent in human language
corpora, but they also inherit inaccuracies and biases present in the data they are
trained on. Notable examples include OpenAI's GPT models, Google's PaLM and Gemini.""",

    """Speculative decoding is a technique used to accelerate the inference of large
language models. The basic idea is to use a smaller, faster draft model to generate
several candidate tokens, which are then verified in parallel by the larger target
model. If the candidates match what the target model would have generated, multiple
tokens can be accepted in a single forward pass, achieving speedup. The key insight
is that the draft model only needs to be accurate enough to predict tokens that the
target model would also predict. When predictions diverge, the speculation is rejected
and only the accepted prefix is kept. This approach exploits the fact that for many
tokens, especially common words and predictable patterns, even a much smaller model
can correctly predict what the larger model would generate. EAGLE is one such method
that uses the hidden states of the target model to guide a lightweight draft head.""",
]


def load_model_and_data(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {cfg.model_name}...")
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, dtype=cfg.dtype,
    ).cuda()
    model.eval()
    print(f"  Layers: {model.config.num_hidden_layers}, "
          f"KV heads: {getattr(model.config, 'num_key_value_heads', '?')}, "
          f"Attn impl: {model.config._attn_implementation}")

    # Tokenize hardcoded passages, pad/truncate to required length
    chunk_size = cfg.prefix_len + max(cfg.draft_lengths) + 10
    samples = []
    for i in range(cfg.num_samples):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        tokens = tok.encode(text, add_special_tokens=True)
        if len(tokens) >= chunk_size:
            tokens = tokens[:chunk_size]
        else:
            # Repeat the text to fill
            while len(tokens) < chunk_size:
                tokens = tokens + tok.encode(text, add_special_tokens=False)
            tokens = tokens[:chunk_size]
        samples.append(torch.tensor(tokens).unsqueeze(0))
    print(f"  Prepared {len(samples)} samples (each {chunk_size} tokens)")
    return model, tok, samples


def run_all():
    print("=" * 60)
    print("Phase 2A: SVC + EAGLE-style Speculation")
    print("=" * 60)

    model, tok, samples = load_model_and_data(cfg)

    mem = analyze_memory_budget(model, cfg)
    save_json(mem, 'memory_budget')

    t0 = time.time()
    r1 = exp2a1_hidden_state_quality(model, samples, cfg)
    save_json(r1, 'exp2a1_hidden_quality')
    print(f"  [exp 2A.1 took {time.time() - t0:.1f}s]")

    t0 = time.time()
    r2 = exp2a2_draft_acceptance(model, samples, cfg)
    save_json(r2, 'exp2a2_acceptance')
    print(f"  [exp 2A.2 took {time.time() - t0:.1f}s]")

    t0 = time.time()
    r3 = exp2a3_wide_tree(model, samples, cfg)
    save_json(r3, 'exp2a3_wide_tree')
    print(f"  [exp 2A.3 took {time.time() - t0:.1f}s]")

    # Final summary
    print("\n" + "=" * 60)
    print("PHASE 2A SUMMARY")
    print("=" * 60)
    s = r1['summary']
    print(f"\n1. Hidden State Quality:")
    print(f"   cos={s['cos_sim_mean']:.6f}  L2_rel={s['l2_rel_mean']:.6f}  "
          f"KL={s['kl_mean']:.6f}  Top1={s['top1_rate']:.0%}")

    print(f"\n2. Draft Acceptance:")
    for g in sorted(r2.keys()):
        rr = r2[g]
        print(f"   γ={g:2d}: per-token={rr['mean_acceptance_rate']:.1%}, "
              f"full={rr['full_accept_rate']:.0%}")

    print(f"\n3. Wide Tree:")
    for w in sorted(r3.keys()):
        rr = r3[w]
        print(f"   W={w:3d}: P(any_full)={rr['any_full_match_rate']:.0%}, "
              f"best={rr['mean_best_match']:.1f}/4")

    print(f"\n4. Memory Budget:")
    print(f"   Savings: {mem['savings_mb']:.1f} MB → {mem['extra_branches']:.0f} extra branches")

    # Go/No-Go
    accept_4 = r2.get(4, {}).get('mean_acceptance_rate', 0)
    cos = s['cos_sim_mean']
    print(f"\n   Decision:")
    if cos > 0.999 and accept_4 > 0.9:
        print(f"     ✓ STRONG GO: cos={cos:.4f}, γ=4 acc={accept_4:.0%}")
    elif cos > 0.99 and accept_4 > 0.7:
        print(f"     ~ CONDITIONAL GO: cos={cos:.4f}, γ=4 acc={accept_4:.0%}")
        print(f"       → QA-training likely needed")
    else:
        print(f"     ✗ Quantization too aggressive: cos={cos:.4f}, γ=4 acc={accept_4:.0%}")

    print(f"\nResults: {cfg.output_dir}/")


if __name__ == "__main__":
    run_all()
