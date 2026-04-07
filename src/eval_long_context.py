"""
Long-context stress test for SVC.

The chunked DPCM encoding has independent errors per chunk → in theory the
quality should not degrade with sequence length. But the COMPOUND effect of
many small per-chunk errors over a long prefix might still affect downstream
hidden states differently than at L=512.

This experiment varies prefix length and measures acceptance rate at γ=4, 8.
"""

import os
import json
import time
import torch
import torch.nn.functional as F
import numpy as np

from svc_kv_cache import DPCMCodec
from phase2a_experiment import (
    svc_quantize_cache, extract_kv_tensors, build_cache,
)


# Use diverse non-repeated passages concatenated. Real long-context data is
# more representative than a single passage repeated many times (the latter
# creates degenerate attention patterns that aren't typical of real workloads).
_PASSAGES = [
    "The history of the universe begins with the Big Bang, an event approximately 13.8 billion years ago. In the first fraction of a second, the universe underwent rapid inflation, expanding exponentially. After this inflationary epoch, the universe was filled with a hot, dense plasma of particles. As it cooled, fundamental forces separated and elementary particles formed. The first atomic nuclei were created during nucleosynthesis, primarily hydrogen and helium with traces of lithium.",
    "Photosynthesis is the biological process by which plants, algae, and certain bacteria convert light energy from the sun into chemical energy stored in glucose. This process occurs in chloroplasts, organelles containing the green pigment chlorophyll. The overall reaction can be summarized as carbon dioxide and water reacting in the presence of sunlight to produce glucose and oxygen. Photosynthesis is foundational to most life on Earth, providing both the oxygen we breathe and the energy that flows through ecosystems.",
    "The Roman Empire reached its greatest territorial extent under Emperor Trajan in the early second century. At its peak, it stretched from Britain in the west to Mesopotamia in the east, encompassing much of Europe, North Africa, and the Middle East. Roman engineering, law, language, and culture left lasting legacies on Western civilization. The empire eventually fragmented under pressures from internal strife, economic difficulties, and external invasions, with the western half collapsing in the fifth century.",
    "Quantum mechanics is the branch of physics that describes the behavior of matter and energy at the smallest scales. Unlike classical mechanics, quantum theory predicts probabilities rather than deterministic outcomes. Key concepts include wave-particle duality, the uncertainty principle, and superposition. Quantum mechanics underlies modern technologies including transistors, lasers, and medical imaging, and forms the basis for emerging fields such as quantum computing and quantum cryptography.",
    "The water cycle, also known as the hydrologic cycle, describes the continuous movement of water on, above, and below the surface of the Earth. Water evaporates from oceans, lakes, and rivers, rises into the atmosphere, condenses into clouds, and falls back as precipitation. Some water flows over land as runoff, while other water infiltrates the ground to become groundwater. This cycle redistributes water and energy across the planet and is essential for life and weather patterns.",
    "Artificial neural networks are computing systems loosely inspired by the biological neural networks that make up animal brains. They consist of interconnected nodes called neurons organized into layers. Each connection has a weight that adjusts during training as the network learns to perform tasks. Modern deep learning uses neural networks with many layers, enabling breakthroughs in image recognition, natural language processing, and game playing. Training requires large datasets and significant computational resources.",
    "The French Revolution began in 1789 and dramatically transformed France and influenced the entire Western world. It overthrew the monarchy, established a republic, and challenged the privileges of the aristocracy and clergy. The revolution unfolded in several phases marked by both democratic reforms and violent upheavals such as the Reign of Terror. It ultimately gave rise to the rule of Napoleon Bonaparte, whose military campaigns spread revolutionary ideas across Europe even as he established his own empire.",
    "Plate tectonics is the scientific theory that explains the large-scale motions of Earth's outer crust. The lithosphere is divided into rigid plates that float on the underlying semi-fluid asthenosphere. These plates move at rates of a few centimeters per year, driven by convection currents in the mantle. Plate boundaries are zones of significant geological activity, including earthquakes, volcanic eruptions, and the formation of mountain ranges. The theory unifies many observations in geology and geophysics.",
]
LONG_TEXT = " ".join(_PASSAGES * 6)  # ~6000+ tokens, diverse but bounded


def measure_long_acceptance(model, tok, prefix_len, gamma, num_starts, device):
    inputs = tok(LONG_TEXT, return_tensors="pt", truncation=True, max_length=prefix_len).to(device)
    input_ids = inputs.input_ids
    actual_len = input_ids.shape[1]

    # Prefill
    out = model(input_ids, use_cache=True)
    kv_full_t = extract_kv_tensors(out.past_key_values)
    kv_quant_t = svc_quantize_cache(kv_full_t, 3, 64)
    del out
    torch.cuda.empty_cache()

    # Reference: 64 tokens of greedy continuation with full cache
    ref_tokens = []
    ref_cache = build_cache(kv_full_t)
    out2 = model(input_ids[:, -1:], past_key_values=ref_cache, use_cache=True)
    cur = out2.logits[:, -1:, :].argmax(dim=-1)
    ref_cache = out2.past_key_values
    del out2
    for _ in range(64):
        ref_tokens.append(cur[0, 0].item())
        v_out = model(cur, past_key_values=ref_cache, use_cache=True)
        ref_cache = v_out.past_key_values
        cur = v_out.logits[:, -1:, :].argmax(dim=-1)
    del ref_cache, v_out
    torch.cuda.empty_cache()

    # For each starting offset, draft γ from quantized cache, compare to ref
    accepts = []
    starts = list(range(0, max(1, 64 - gamma), max(1, (64 - gamma) // max(num_starts, 1))))[:num_starts]
    for start in starts:
        draft_cache = build_cache(kv_quant_t)
        if start > 0:
            ext = torch.tensor([ref_tokens[:start]], device=device)
            d0 = model(ext, past_key_values=draft_cache, use_cache=True)
            draft_cache = d0.past_key_values
            del d0

        cur_d = torch.tensor([[ref_tokens[start]]], device=device)
        d_out = model(cur_d, past_key_values=draft_cache, use_cache=True)
        draft_cache = d_out.past_key_values
        cur_d = d_out.logits[:, -1:, :].argmax(dim=-1)
        draft_seq = []
        for _ in range(gamma):
            draft_seq.append(cur_d[0, 0].item())
            d_out = model(cur_d, past_key_values=draft_cache, use_cache=True)
            draft_cache = d_out.past_key_values
            cur_d = d_out.logits[:, -1:, :].argmax(dim=-1)

        target = ref_tokens[start + 1: start + 1 + gamma]
        n = 0
        for j in range(min(gamma, len(target))):
            if draft_seq[j] == target[j]:
                n += 1
            else:
                break
        accepts.append(n)
        del draft_cache, d_out
        torch.cuda.empty_cache()

    del kv_full_t, kv_quant_t
    torch.cuda.empty_cache()
    return accepts, actual_len


def run():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = os.environ.get("SVC_MODEL", "Qwen/Qwen2.5-7B")
    print(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16,
    ).cuda()
    model.eval()
    device = "cuda"

    out_dir = os.environ.get("SVC_OUTDIR", "./long_context_results")
    os.makedirs(out_dir, exist_ok=True)

    prefix_lens = [256, 512, 1024, 2048, 3072]
    gammas = [4, 8]

    results = {}
    for L in prefix_lens:
        results[L] = {}
        for g in gammas:
            t0 = time.time()
            accepts, actual = measure_long_acceptance(model, tok, L, g, 8, device)
            full_match = sum(1 for a in accepts if a == g) / len(accepts)
            mean = float(np.mean(accepts))
            results[L][g] = {
                'actual_prefix': actual,
                'mean_accepted': mean,
                'per_token': mean / g,
                'full_match': full_match,
                'distribution': dict(__import__('collections').Counter(accepts)),
            }
            print(f"  L={L:4d} (actual {actual}), γ={g}: "
                  f"mean={mean:.2f}/{g}, per-tok={mean/g:.1%}, full={full_match:.1%} "
                  f"[{time.time()-t0:.0f}s]")
            torch.cuda.empty_cache()

    with open(f"{out_dir}/long_context.json", "w") as f:
        out_data = {str(k): {str(g): v for g, v in d.items()} for k, d in results.items()}
        json.dump(out_data, f, indent=2)
    print(f"\nSaved to {out_dir}/long_context.json")


if __name__ == "__main__":
    run()
