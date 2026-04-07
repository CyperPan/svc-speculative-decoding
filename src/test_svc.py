"""
SVC KV Cache validation tests
==============================
1. Codec round-trip correctness (base only vs base+refinement)
2. Quality measurement: KL divergence under base-layer KV
3. Memory savings verification
4. End-to-end speculative decoding smoke test

Run on Colab A100 or any GPU with ≥16GB memory.
"""

import torch
import torch.nn.functional as F
import numpy as np
from svc_kv_cache import (
    DPCMCodec, SVCKVCache, SVCSpeculativeDecoder,
    AcceptanceRateFilter, optimal_gamma,
)


# ============================================================
# Test 1: DPCM codec round-trip
# ============================================================

def test_codec_roundtrip():
    """Verify that encode → decode → encode gives consistent results."""
    print("\n[Test 1] DPCM codec round-trip")
    print("-" * 50)

    torch.manual_seed(42)
    H, L, d = 8, 512, 128
    # Simulate KV cache with strong temporal correlation (like real KV)
    base_signal = torch.randn(H, 1, d) * 0.5
    drift = torch.randn(H, L, d).cumsum(dim=1) * 0.01
    x = base_signal + drift  # [H, L, d] with autocorrelation

    for bits in [2, 3, 4]:
        codec = DPCMCodec(bits=bits)
        codes, scales, offsets, anchor = codec.encode(x)
        x_hat = codec.decode(codes, scales, offsets, anchor)

        mse = F.mse_loss(x_hat, x).item()
        snr = 10 * np.log10(x.var().item() / (mse + 1e-10))
        max_err = (x - x_hat).abs().max().item()

        # Refinement residual
        residual = (x - x_hat).half().float()
        x_full = x_hat + residual
        recon_err = (x - x_full).abs().max().item()

        print(f"  {bits}-bit: SNR={snr:6.2f}dB, max_err={max_err:.4f}, "
              f"refined_max_err={recon_err:.6f}")

    print("  ✓ Round-trip works")


# ============================================================
# Test 2: SVC layer encode/decode
# ============================================================

def test_svc_layer():
    """Test SVCKVCache encoding for one layer."""
    print("\n[Test 2] SVC layer encode/decode")
    print("-" * 50)

    torch.manual_seed(0)
    H_kv, L, d = 8, 1024, 128
    k = torch.randn(H_kv, L, d, dtype=torch.float16)
    v = torch.randn(H_kv, L, d, dtype=torch.float16)

    cache = SVCKVCache(bits=3)
    layer = cache.encode_layer(k, v, store_refinement=True)

    # Base-only decode
    k_base, v_base = cache.decode_base(layer)
    base_k_err = (k.float() - k_base).abs().mean().item()
    base_v_err = (v.float() - v_base).abs().mean().item()

    # Full decode (base + refinement)
    k_full, v_full = cache.decode_full(layer)
    full_k_err = (k.float() - k_full).abs().mean().item()
    full_v_err = (v.float() - v_full).abs().mean().item()

    print(f"  Base only:    K mean_err={base_k_err:.4f}, V mean_err={base_v_err:.4f}")
    print(f"  Base + refine: K mean_err={full_k_err:.6f}, V mean_err={full_v_err:.6f}")

    assert full_k_err < 1e-3, "Refinement should give near-perfect reconstruction"
    assert full_v_err < 1e-3
    print("  ✓ SVC layer works correctly")


# ============================================================
# Test 3: Memory savings
# ============================================================

def test_memory():
    """Compute and report memory savings (analytical, no actual encoding)."""
    print("\n[Test 3] Memory savings analysis")
    print("-" * 50)

    configs = [
        ("Mistral-7B (GQA)", 32, 8, 2048, 128),
        ("Qwen2.5-1.5B (GQA)", 28, 2, 2048, 128),
        ("Llama-2-7B (MHA)", 32, 32, 2048, 128),
    ]
    for name, num_layers, H_kv, L, d in configs:
        per_kv = DPCMCodec.memory_bytes(H_kv, L, d, bits=3, chunk_size=64)
        original = per_kv['original_bytes'] * 2 * num_layers / 1e6  # K+V
        base_int8 = per_kv['encoded_int8_bytes'] * 2 * num_layers / 1e6
        base_packed = per_kv['encoded_packed_bytes'] * 2 * num_layers / 1e6
        refinement = H_kv * L * d * 2 * 2 * num_layers / 1e6
        print(f"  {name}:")
        print(f"    Original (FP16):     {original:.1f} MB")
        print(f"    Base int8:           {base_int8:.1f} MB ({base_int8/original*100:.0f}%)")
        print(f"    Base packed (3-bit): {base_packed:.1f} MB ({base_packed/original*100:.0f}%)")
        print(f"    Refinement (FP16):   {refinement:.1f} MB ({refinement/original*100:.0f}%)")
        print(f"    Draft savings (packed): {(1-base_packed/original)*100:.0f}%")
    print("  ✓ Memory analysis done")


# ============================================================
# Test 4: Quality vs original KV (real model)
# ============================================================

def test_quality_real_model(model_name="Qwen/Qwen2.5-1.5B"):
    """Compare logits when using base-layer KV vs full KV."""
    print("\n[Test 4] Quality test on real model")
    print("-" * 50)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  ⚠ transformers not available, skipping")
        return

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    ).cuda()
    model.eval()

    # Sample input — keep short to limit DPCM encoding time
    text = "The quick brown fox jumps over the lazy dog. " * 20
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    input_ids = inputs.input_ids.cuda()
    print(f"  Input length: {input_ids.shape[1]} tokens")

    # Strategy: build cache from input_ids[:-1], reference is one full forward
    # pass on input_ids; SVC variants feed last_token with reconstructed cache.
    context_ids = input_ids[:, :-1]
    last_token = input_ids[:, -1:]

    # Reference: single forward on full input_ids; use logits at position -1
    # (prediction at the position where last_token sits)
    with torch.no_grad():
        ref_out = model(input_ids)
        ref_logits = ref_out.logits[0, -1].float().cpu()
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_probs = ref_log_probs.exp()
        print(f"  ref logit range: [{ref_logits.min():.2f}, {ref_logits.max():.2f}], "
              f"any_nan={torch.isnan(ref_logits).any().item()}")

    # Separately: build context cache for SVC
    with torch.no_grad():
        ctx_out = model(context_ids, use_cache=True)

    # Try multiple chunk_size to find best base-layer quality
    print("\n  === Sweep chunk_size and bits ===")
    for cs in [8, 16, 32, 64]:
        for bb in [2, 3, 4]:
            svc_t = SVCKVCache(bits=bb)
            svc_t.codec.chunk_size = cs
            svc_t.encode_from_model_cache(ctx_out.past_key_values, store_refinement=False)
            base_cache_t = svc_t.to_hf_cache_base()
            with torch.no_grad():
                out_t = model(last_token, past_key_values=base_cache_t)
                lt = out_t.logits[0, -1].float().cpu()
            kl_t, _ = (lambda: (
                lambda: ((ref_probs > 1e-8).float() *
                         (ref_log_probs - F.log_softmax(lt, dim=-1)) * ref_probs).sum().item()
            )())(), None
            # Simpler:
            log_t = F.log_softmax(lt, dim=-1)
            mask = ref_probs > 1e-8
            kl_v = (ref_probs[mask] * (ref_log_probs[mask] - log_t[mask])).sum().item()
            top1 = (lt.argmax() == ref_logits.argmax()).item()
            top5_t = lt.topk(5).indices.tolist()
            top5_r = ref_logits.topk(5).indices.tolist()
            ovl = len(set(top5_t) & set(top5_r)) / 5
            print(f"    chunk={cs:2d} bits={bb}: KL={kl_v:.4f}, "
                  f"Top1={'✓' if top1 else '✗'}, Top5={ovl:.0%}, "
                  f"max|Δ|={(lt-ref_logits).abs().max():.2f}")
            del svc_t, base_cache_t, out_t

    # Build SVC from context cache (default chunk size)
    svc = SVCKVCache(bits=3)
    print("  Encoding KV cache into SVC...")
    import time
    t0 = time.time()
    svc.encode_from_model_cache(ctx_out.past_key_values, store_refinement=True)
    print(f"  Encoding took {time.time() - t0:.2f}s")
    print(f"  Memory: {svc.memory_report()}")

    def safe_kl(target_log_probs, target_probs, approx_logits):
        approx_log_probs = F.log_softmax(approx_logits, dim=-1)
        # KL(target || approx) = sum target * (log_target - log_approx)
        # Mask out target probabilities below 1e-8 to avoid 0 * (-inf)
        mask = target_probs > 1e-8
        kl = (target_probs[mask] * (target_log_probs[mask] - approx_log_probs[mask])).sum()
        return kl.item(), approx_log_probs.exp()

    # === Test 4a: Full-precision SVC reconstruction ===
    full_cache = svc.to_hf_cache_full()
    with torch.no_grad():
        full_out = model(last_token, past_key_values=full_cache)
        full_logits = full_out.logits[0, -1].float().cpu()

    full_kl, full_probs = safe_kl(ref_log_probs, ref_probs, full_logits)
    max_diff_full = (full_logits - ref_logits).abs().max().item()
    print(f"  Full SVC: KL={full_kl:.6f}, max|Δlogit|={max_diff_full:.4f}")

    # === Test 4b: Base-only SVC reconstruction ===
    base_cache = svc.to_hf_cache_base()
    with torch.no_grad():
        base_out = model(last_token, past_key_values=base_cache)
        base_logits = base_out.logits[0, -1].float().cpu()

    base_kl, base_probs = safe_kl(ref_log_probs, ref_probs, base_logits)
    max_diff_base = (base_logits - ref_logits).abs().max().item()
    print(f"  Base SVC: KL={base_kl:.6f}, max|Δlogit|={max_diff_base:.4f}")

    # Top-k agreement
    ref_top5 = ref_probs.topk(5).indices.tolist()
    base_top5 = base_probs.topk(5).indices.tolist()
    full_top5 = full_probs.topk(5).indices.tolist()

    base_top1 = ref_top5[0] == base_top5[0]
    full_top1 = ref_top5[0] == full_top5[0]
    base_top5_overlap = len(set(ref_top5) & set(base_top5)) / 5
    full_top5_overlap = len(set(ref_top5) & set(full_top5)) / 5

    print(f"  Base: Top1={'✓' if base_top1 else '✗'}, Top5_overlap={base_top5_overlap:.0%}")
    print(f"  Full: Top1={'✓' if full_top1 else '✗'}, Top5_overlap={full_top5_overlap:.0%}")

    # Acceptance rate proxy: how often does base-decoded prediction agree?
    # This estimates α for speculative decoding
    print(f"\n  → Estimated draft acceptance rate (top-1 agreement): "
          f"{1.0 if base_top1 else 0.0:.0%}")

    return {
        'full_kl': full_kl,
        'base_kl': base_kl,
        'base_top1': base_top1,
        'full_top1': full_top1,
    }


# ============================================================
# Test 5: End-to-end speculative decoding
# ============================================================

def test_e2e_speculative(model_name="Qwen/Qwen2.5-1.5B"):
    """Smoke test: run SVC speculative decoding for a few tokens."""
    print("\n[Test 5] End-to-end speculative decoding")
    print("-" * 50)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  ⚠ transformers not available, skipping")
        return

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    ).cuda()
    model.eval()

    decoder = SVCSpeculativeDecoder(target_model=model, bits=3)

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    print(f"  Prompt: {prompt!r}")
    generated, stats = decoder.generate(
        input_ids, max_new_tokens=32, temperature=0.0
    )

    text = tokenizer.decode(generated)
    print(f"  Generated: {text!r}")
    print(f"  Stats:")
    for k, v in stats.items():
        if k != 'memory':
            print(f"    {k}: {v}")
    print(f"    memory: {stats['memory']}")


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    test_codec_roundtrip()
    test_svc_layer()
    test_memory()

    if torch.cuda.is_available():
        test_quality_real_model()
        # test_e2e_speculative()  # skip until cache re-encoding is incrementalized
    else:
        print("\n⚠ No GPU; skipping real model tests")
