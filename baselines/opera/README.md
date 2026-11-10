# OPERA baseline (ported implementation)

Paper baseline "OPERA: Over-Trast Penalty and Retrospection-Allocation" (Huang et al., CVPR 2024).
The official OPERA release binds to transformers 4.29 / LLaVA / MiniGPT-4; for the Qwen2.5-VL
backbone used in the SkinCD paper we used a faithful independent port:

- `opera_qwen.py` — partial autoregressive candidate scoring, over-trust penalty
  (first 10 steps, image-token aggregate attention), and rollback re-decoding.
  Paper-default parameters: scale_factor=50, threshold=15, num_attn_candidates=5,
  penalty_weights=1.0, rollback_budget=3.
- Invoked from an internal evaluation harness (`ZS_classification.py --method opera`,
  free-generation protocol, greedy) — that harness is not part of this release;
  wiring `opera_generate_qwen()` into `experiments/eval/classification.py` is ~10 lines
  (replace the `model.generate` call), see the call signature in the module docstring.
- `opera_reference_results.json` — reference numbers from our runs (free-generation
  protocol, hence not directly comparable to the label-constrained main table).

## Headline reference numbers (accuracy, n)

| dataset | OPERA | base (same protocol) |
|---|---:|---:|
| HAM10000 (1512) | 29.56% | 32.80% (−3.24) |
| Patch16 (640, 16-class subset) | 16.88% | 27.50% (−10.62) |
| PAD-UFES-20 (681) | 48.16% | 49.26% (−1.10) |

Consistent with the paper's main-table finding: OPERA underperforms SkinCD on skin
disease classification across all three modalities.
