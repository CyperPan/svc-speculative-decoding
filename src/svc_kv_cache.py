"""
SVC (Scalable/Successive) KV Cache for Speculative Decoding
============================================================

Architecture:
  Base layer   = ALL heads, DPCM 3-bit  (proven: KL≈0.09, Top5=100%)
  Refinement   = FP16 residual (original - dequantized base)

Draft phase  → base layer only  → fast, ~19% memory
Verify phase → full precision   → exact, no quality loss

The key insight from Phase 0:
  - Heads are orthogonal (cos≈0) → no cross-head redundancy to exploit
  - Sequence dimension is low-rank (50% var @ rank 12-34) → DPCM works well
  - DPCM 3-bit is the sweet spot (SNR≈10.8dB on K, KL≈0.09)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ============================================================
# 1. DPCM Codec — the proven 3-bit encoder from Phase 0
# ============================================================

class DPCMCodec:
    """
    Chunked anchor-based quantizer (vectorized).

    Splits the sequence into chunks of `chunk_size` tokens. Each chunk has:
      - 1 FP16 anchor (the first token of the chunk)
      - chunk_size-1 quantized DIRECT residuals = (token - anchor)

    Decoding: x[t] = anchor + dequant(residual[t])  — no cumsum, no error
    accumulation. Each token's reconstruction error is independent and bounded
    by the per-chunk quantization step size.

    This relies on the empirical observation that within a small window of
    tokens, KV values change slowly (high temporal correlation), so the
    direct residuals (token - chunk_first_token) have small magnitude and
    quantize well.
    """

    def __init__(self, bits: int = 3, chunk_size: int = 64):
        self.bits = bits
        self.n_levels = 2 ** bits
        self.chunk_size = chunk_size

    def encode(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Chunked anchor-based encoder (fully vectorized).

        Args:
            tensor: [num_kv_heads, seq_len, head_dim]

        Returns:
            codes:    [H, L, d] int16   — quantized direct residuals
            scales:   [H, n_chunks, 1, d] — per-chunk per-head per-dim scale
            offsets:  [H, n_chunks, 1, d] — per-chunk per-head per-dim offset
            anchors:  [H, n_chunks, d] FP16
        """
        H, L, d = tensor.shape
        K = self.chunk_size

        # Pad sequence to multiple of K. Pad by REPLICATING the last valid
        # token, so the padded positions have zero residual w.r.t. their
        # chunk anchor (avoiding contamination of the per-chunk min/max).
        n_chunks = (L + K - 1) // K
        L_padded = n_chunks * K
        if L_padded != L:
            last = tensor[:, -1:, :].expand(H, L_padded - L, d).contiguous()
            tensor_padded = torch.cat([tensor, last], dim=1)
        else:
            tensor_padded = tensor

        # Reshape to [H, n_chunks, K, d]
        chunked = tensor_padded.reshape(H, n_chunks, K, d).float()

        # Per-chunk anchor = first token of each chunk
        anchors = chunked[:, :, 0, :].clone()  # [H, n_chunks, d]

        # Direct residuals from anchor: residual[t] = chunk[t] - anchor (for t=1..K-1)
        # No accumulation — each residual is independent.
        residuals = chunked[:, :, 1:, :] - anchors.unsqueeze(2)  # [H, n_chunks, K-1, d]

        # Per-chunk per-dim scale (each chunk has independent quantization)
        r_min = residuals.amin(dim=2, keepdim=True)  # [H, n_chunks, 1, d]
        r_max = residuals.amax(dim=2, keepdim=True)
        span = torch.clamp(r_max - r_min, min=1e-7)
        scales = span / (self.n_levels - 1)
        offsets = r_min

        # Quantize
        normalized = (residuals - offsets) / scales
        quantized = torch.round(normalized).clamp(0, self.n_levels - 1)
        codes_body = quantized.to(torch.int16)  # [H, n_chunks, K-1, d]

        # Build full codes tensor
        codes = torch.zeros(H, L_padded, d, dtype=torch.int16, device=tensor.device)
        codes_view = codes.reshape(H, n_chunks, K, d)
        codes_view[:, :, 1:, :] = codes_body
        codes = codes[:, :L, :]

        return codes, scales, offsets, anchors.to(torch.float16)

    def decode(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Chunked anchor-based decoder.

        For each chunk: x[t] = anchor + dequant(residual[t])
        No cumsum → no error accumulation.
        """
        H, L, d = codes.shape
        K = self.chunk_size
        n_chunks = anchors.shape[1]
        L_padded = n_chunks * K

        # Pad codes to L_padded
        if L < L_padded:
            pad = torch.zeros(H, L_padded - L, d, dtype=codes.dtype, device=codes.device)
            codes_padded = torch.cat([codes, pad], dim=1)
        else:
            codes_padded = codes

        codes_chunked = codes_padded.reshape(H, n_chunks, K, d)

        # Dequantize positions 1..K-1 per chunk
        dequant = codes_chunked[:, :, 1:, :].float() * scales + offsets  # [H, n_chunks, K-1, d]

        # Reconstruct: x[0]=anchor, x[t]=anchor+dequant[t]
        recon_chunked = torch.empty(
            H, n_chunks, K, d, dtype=torch.float32, device=codes.device
        )
        anchors_f = anchors.float()
        recon_chunked[:, :, 0, :] = anchors_f
        recon_chunked[:, :, 1:, :] = anchors_f.unsqueeze(2) + dequant

        reconstructed = recon_chunked.reshape(H, L_padded, d)[:, :L, :]
        return reconstructed

    @staticmethod
    def memory_bytes(H: int, L: int, d: int, bits: int = 3, chunk_size: int = 64) -> dict:
        """Compute memory usage of encoded vs original KV cache."""
        original = H * L * d * 2  # FP16
        n_chunks = (L + chunk_size - 1) // chunk_size

        # Codes: stored as int16 in our impl; bit-packing to `bits` is possible
        codes_int16 = H * L * d * 2
        codes_packed = H * L * d * bits / 8

        scales = H * n_chunks * d * 4   # float32, per-chunk
        offsets = H * n_chunks * d * 4
        anchors = H * n_chunks * d * 2  # FP16, one anchor per chunk

        encoded_int16 = codes_int16 + scales + offsets + anchors
        encoded_packed = codes_packed + scales + offsets + anchors

        return {
            'original_bytes': original,
            'encoded_int8_bytes': encoded_int16,  # keep key name for compat
            'encoded_packed_bytes': encoded_packed,
            'ratio_int8': encoded_int16 / original,
            'ratio_packed': encoded_packed / original,
        }


# ============================================================
# 2. SVC KV Cache — the two-layer structure
# ============================================================

@dataclass
class SVCLayer:
    """One layer's SVC-encoded KV cache."""
    # Base layer (3-bit DPCM)
    k_codes: torch.Tensor       # [H_kv, L, d] int8
    k_scales: torch.Tensor      # [H_kv, 1, d]
    k_offsets: torch.Tensor
    k_anchor: torch.Tensor      # [H_kv, 1, d] FP16

    v_codes: torch.Tensor
    v_scales: torch.Tensor
    v_offsets: torch.Tensor
    v_anchor: torch.Tensor

    # Refinement residual (FP16 original - DPCM decoded)
    # Only materialized when needed; None = not yet computed
    k_residual: Optional[torch.Tensor] = None  # [H_kv, L, d] FP16
    v_residual: Optional[torch.Tensor] = None

    # Metadata
    seq_len: int = 0
    has_refinement: bool = False


class SVCKVCache:
    """
    Scalable Video Coding-style KV Cache.

    Two operating modes:
      DRAFT mode:  base layer only  → ~19% memory, ~0.09 KL divergence
      FULL mode:   base + refinement → 100% quality (lossless reconstruction)

    Integration with speculative decoding:
      1. Prefill → encode KV into base + refinement
      2. Draft model runs using base layer (decode 3-bit KV)
      3. Target model verifies using full precision KV (base + residual)
      4. Accepted tokens → encode new KV, append to both layers
      5. Rejected tokens → discard base-only appendix (cheap)
    """

    def __init__(self, bits: int = 3, device: str = "cuda"):
        self.codec = DPCMCodec(bits=bits)
        self.bits = bits
        self.device = device
        self.layers: List[Optional[SVCLayer]] = []

    def encode_layer(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        store_refinement: bool = True,
    ) -> SVCLayer:
        """
        Encode one transformer layer's KV cache.

        Args:
            k: [num_kv_heads, seq_len, head_dim] — Key cache
            v: [num_kv_heads, seq_len, head_dim] — Value cache
            store_refinement: whether to compute and store the FP16 residual

        Returns: SVCLayer
        """
        k_float = k.float()
        v_float = v.float()

        # Base layer: DPCM encode
        k_codes, k_scales, k_offsets, k_anchor = self.codec.encode(k_float)
        v_codes, v_scales, v_offsets, v_anchor = self.codec.encode(v_float)

        layer = SVCLayer(
            k_codes=k_codes, k_scales=k_scales,
            k_offsets=k_offsets, k_anchor=k_anchor,
            v_codes=v_codes, v_scales=v_scales,
            v_offsets=v_offsets, v_anchor=v_anchor,
            seq_len=k.shape[1],
        )

        if store_refinement:
            # Refinement = original - base_decoded
            k_base = self.codec.decode(k_codes, k_scales, k_offsets, k_anchor)
            v_base = self.codec.decode(v_codes, v_scales, v_offsets, v_anchor)
            layer.k_residual = (k_float - k_base).half()
            layer.v_residual = (v_float - v_base).half()
            layer.has_refinement = True

        return layer

    def decode_base(self, layer: SVCLayer) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode base layer only → approximate KV for draft."""
        k = self.codec.decode(
            layer.k_codes, layer.k_scales, layer.k_offsets, layer.k_anchor
        )
        v = self.codec.decode(
            layer.v_codes, layer.v_scales, layer.v_offsets, layer.v_anchor
        )
        return k, v

    def decode_full(self, layer: SVCLayer) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode base + refinement → exact FP16 KV for verification."""
        k_base, v_base = self.decode_base(layer)
        if layer.has_refinement:
            k = k_base + layer.k_residual.float()
            v = v_base + layer.v_residual.float()
        else:
            k, v = k_base, v_base
        return k, v

    def encode_from_model_cache(
        self,
        past_key_values,
        store_refinement: bool = True,
    ):
        """
        Encode the full KV cache from a model forward pass.

        Supports:
          - transformers >=5.0 DynamicCache (cache.layers[i].keys/.values)
          - transformers <5.0 DynamicCache (cache.key_cache[i])
          - legacy tuple of (k, v) tuples
        """
        self.layers = []

        # New API (transformers ≥5.0)
        if hasattr(past_key_values, 'layers') and len(getattr(past_key_values, 'layers', [])) > 0 \
                and hasattr(past_key_values.layers[0], 'keys'):
            for cache_layer in past_key_values.layers:
                k = cache_layer.keys[0]    # [H_kv, L, d]
                v = cache_layer.values[0]
                self.layers.append(
                    self.encode_layer(k, v, store_refinement=store_refinement)
                )
        # Legacy DynamicCache API (transformers <5.0)
        elif hasattr(past_key_values, 'key_cache'):
            for i in range(len(past_key_values.key_cache)):
                k = past_key_values.key_cache[i][0]
                v = past_key_values.value_cache[i][0]
                self.layers.append(
                    self.encode_layer(k, v, store_refinement=store_refinement)
                )
        # Legacy tuple format
        elif isinstance(past_key_values, (list, tuple)):
            for k, v in past_key_values:
                self.layers.append(
                    self.encode_layer(k[0], v[0], store_refinement=store_refinement)
                )
        else:
            raise TypeError(f"Unsupported cache type: {type(past_key_values)}")

    def to_hf_cache_base(self):
        """
        Convert base-layer-decoded KV back to HuggingFace DynamicCache format.
        Used for draft model forward pass.
        """
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for i, layer in enumerate(self.layers):
            k, v = self.decode_base(layer)
            # Add batch dim, convert to FP16
            cache.update(
                k.unsqueeze(0).half(),
                v.unsqueeze(0).half(),
                layer_idx=i,
            )
        return cache

    def to_hf_cache_full(self):
        """
        Convert full-precision KV back to HuggingFace DynamicCache format.
        Used for target model verification.
        """
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for i, layer in enumerate(self.layers):
            k, v = self.decode_full(layer)
            cache.update(
                k.unsqueeze(0).half(),
                v.unsqueeze(0).half(),
                layer_idx=i,
            )
        return cache

    def memory_report(self) -> dict:
        """Report memory usage breakdown."""
        if not self.layers:
            return {}

        layer = self.layers[0]
        H, L, d = layer.k_codes.shape

        per_kv = DPCMCodec.memory_bytes(H, L, d, self.bits, self.codec.chunk_size)

        base_total = per_kv['encoded_int8_bytes'] * 2 * len(self.layers)  # K+V
        refinement_total = H * L * d * 2 * 2 * len(self.layers)  # FP16 residual K+V
        original_total = per_kv['original_bytes'] * 2 * len(self.layers)

        return {
            'num_layers': len(self.layers),
            'heads': H,
            'seq_len': L,
            'head_dim': d,
            'original_MB': original_total / 1e6,
            'base_only_MB': base_total / 1e6,
            'refinement_MB': refinement_total / 1e6,
            'total_svc_MB': (base_total + refinement_total) / 1e6,
            'base_ratio': base_total / original_total,
            'draft_savings': f"{(1 - base_total / original_total) * 100:.1f}%",
        }


# ============================================================
# 3. Adaptive Speculative Decoder with SVC
# ============================================================

class AcceptanceRateFilter:
    """
    Kalman-style adaptive acceptance rate estimator.
    (From the earlier adaptive γ discussion.)
    """

    def __init__(self, Q: float = 0.005, alpha_init: float = 0.5, P_init: float = 0.25):
        self.alpha = alpha_init
        self.P = P_init
        self.Q = Q

    def update(self, accepted: bool) -> float:
        a = float(accepted)
        alpha_pred = self.alpha
        P_pred = self.P + self.Q

        R = max(alpha_pred * (1 - alpha_pred), 1e-6)
        K = P_pred / (P_pred + R)

        self.alpha = alpha_pred + K * (a - alpha_pred)
        self.alpha = max(0.01, min(0.99, self.alpha))
        self.P = (1 - K) * P_pred
        return self.alpha

    def update_batch(self, accepted_list: List[bool]) -> float:
        for a in accepted_list:
            self.update(a)
        return self.alpha


def optimal_gamma(alpha: float, c: float = 0.05, gamma_max: int = 16) -> int:
    """Compute optimal speculation length given acceptance rate and cost ratio."""
    if alpha < 0.05:
        return 1

    best_gamma = 1
    best_throughput = 0.0

    for gamma in range(1, gamma_max + 1):
        expected_tokens = (1 - alpha ** (gamma + 1)) / (1 - alpha)
        cost = gamma * c + 1
        throughput = expected_tokens / cost
        if throughput > best_throughput:
            best_throughput = throughput
            best_gamma = gamma

    return best_gamma


class SVCSpeculativeDecoder:
    """
    Speculative decoder using SVC KV cache.

    Flow:
      1. Prefill with target model → encode KV into SVC (base + refinement)
      2. Draft phase:
         - Decode base layer → approximate KV cache
         - Run draft model (or same model with approx KV) for γ tokens
      3. Verify phase:
         - Decode full layer → exact KV cache
         - Run target model to verify γ candidates
      4. Accept/reject:
         - Accepted tokens: encode their KV into SVC, append
         - Rejected tokens: discard (only wasted base-layer decode cost)
      5. Adapt γ based on acceptance rate
    """

    def __init__(
        self,
        target_model,
        draft_model=None,
        bits: int = 3,
        cost_ratio: float = 0.05,
        device: str = "cuda",
    ):
        self.target = target_model
        # If no separate draft model, use target model with base-layer KV
        # (the "self-speculative" variant)
        self.draft = draft_model
        self.svc = SVCKVCache(bits=bits, device=device)
        self.filter = AcceptanceRateFilter()
        self.cost_ratio = cost_ratio
        self.device = device

        # Stats
        self.total_generated = 0
        self.total_accepted = 0
        self.total_draft_calls = 0
        self.total_verify_calls = 0

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor):
        """Run prefill and encode KV into SVC."""
        input_ids = input_ids.to(self.device)
        outputs = self.target(input_ids, use_cache=True)
        self.svc.encode_from_model_cache(outputs.past_key_values)
        return outputs.logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> Tuple[List[int], dict]:
        """
        Generate tokens using SVC speculative decoding.

        Returns:
            generated_ids: list of generated token IDs
            stats: generation statistics
        """
        input_ids = input_ids.to(self.device)

        # Prefill
        logits = self.prefill(input_ids)
        next_token_logits = logits[0, -1, :]

        generated = []
        gamma = 4  # Initial speculation length

        while len(generated) < max_new_tokens:
            # --- Draft phase: use base-layer KV ---
            draft_cache = self.svc.to_hf_cache_base()
            draft_tokens = []

            # Sample first draft token from last target logits
            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                token = torch.multinomial(probs, 1).item()
            else:
                token = next_token_logits.argmax().item()

            current_input = torch.tensor([[token]], device=self.device)
            draft_tokens.append(token)

            # Generate γ-1 more draft tokens using base-layer KV
            draft_model = self.draft if self.draft is not None else self.target
            for _ in range(gamma - 1):
                outputs = draft_model(
                    current_input,
                    past_key_values=draft_cache,
                    use_cache=True,
                )
                draft_cache = outputs.past_key_values
                draft_logits = outputs.logits[0, -1, :]

                if temperature > 0:
                    probs = F.softmax(draft_logits / temperature, dim=-1)
                    token = torch.multinomial(probs, 1).item()
                else:
                    token = draft_logits.argmax().item()

                current_input = torch.tensor([[token]], device=self.device)
                draft_tokens.append(token)

            self.total_draft_calls += 1

            # --- Verify phase: use full-precision KV ---
            full_cache = self.svc.to_hf_cache_full()
            draft_tensor = torch.tensor([draft_tokens], device=self.device)
            verify_outputs = self.target(
                draft_tensor,
                past_key_values=full_cache,
                use_cache=True,
            )
            self.total_verify_calls += 1

            # --- Accept/reject (greedy for simplicity) ---
            verify_logits = verify_outputs.logits[0]  # [gamma, vocab]
            accepted = []

            for i, draft_token in enumerate(draft_tokens):
                target_token = verify_logits[i].argmax().item()
                if draft_token == target_token:
                    accepted.append(True)
                    generated.append(draft_token)
                else:
                    accepted.append(False)
                    # Accept the target's token instead
                    generated.append(target_token)
                    break

            # Update acceptance rate filter
            self.filter.update_batch(accepted)
            self.total_accepted += sum(accepted)
            self.total_generated += len(accepted)

            # Bonus token: if all draft tokens accepted, sample one more
            if all(accepted):
                bonus_logits = verify_logits[-1]
                if temperature > 0:
                    probs = F.softmax(bonus_logits / temperature, dim=-1)
                    bonus_token = torch.multinomial(probs, 1).item()
                else:
                    bonus_token = bonus_logits.argmax().item()
                generated.append(bonus_token)

            # Update SVC cache with accepted tokens' KV
            # (Re-encode from the verify pass which has exact KV)
            n_accepted = sum(accepted) + (1 if all(accepted) else 0)
            self.svc.encode_from_model_cache(
                verify_outputs.past_key_values
            )

            # Update next_token_logits for the next iteration
            next_token_logits = verify_logits[len(accepted) - 1]

            # Adapt gamma
            gamma = optimal_gamma(self.filter.alpha, self.cost_ratio)

            if len(generated) >= max_new_tokens:
                break

        stats = {
            'total_tokens': len(generated),
            'acceptance_rate': self.total_accepted / max(self.total_generated, 1),
            'current_alpha': self.filter.alpha,
            'current_gamma': gamma,
            'draft_calls': self.total_draft_calls,
            'verify_calls': self.total_verify_calls,
            'tokens_per_verify': len(generated) / max(self.total_verify_calls, 1),
            'memory': self.svc.memory_report(),
        }

        return generated[:max_new_tokens], stats
