# Note on the Patch16 protocol (protocol footnote, plan-B draft 2026-11-07)

> 这一文件对应 README §7 的第一行，是脚注的完整展开版。发布时放进 `docs/PATCH16_PROTOCOL_NOTE.md`，README 里链过去即可。三段结构：**发生了什么 → 我们怎么排查的 → 发布口径的正当性**。

## 1. 发生了什么（What）

The Patch16 entry of the paper's main comparison (Base 48.85 → SkinCD 66.42 Accuracy, +17.57 pp) was produced with a test list resolved at runtime from an **internal dataset registry** (`SkinRefocus/utils/dataset_registry.py`, key `patch16`). That registry file resided on the authors' compute node (`/mnt/data/ssz/...`) and was **not part of the anonymous code snapshot**, nor of this release. The snapshot only ships the resolution *calls* (`run_multidataset_dola_vola.sh`), not the *definitions*.

Consequence: the exact image list, label wording, and any subsampling behind 48.85/66.42 **cannot be reconstructed from the released materials**. We state this plainly rather than silently substituting numbers.

## 2. 我们怎么排查的（How we established this）

Full archaeology is in `docs/PROTOCOL_INVESTIGATION.md` (2026-11-06). Summary of the evidence chain:

1. **Integer-constraint analysis.** Reported accuracies have two decimals; for a list of size n, "acc = c/n rounds to the reported value" admits an integer c only for certain n. Across every list available to us — n = 640, 1,941, 2,804, 28,039 (official Test variants) — the paper's four Patch16 numbers (48.85 / 40.10 / 38.03 / 66.42) admit **no** simultaneous integer solution (e.g., 48.85% × 2804 = 1369.75, not an integer). They do admit a simultaneous solution at **n = 12,354 — the official Validation split** (6035/12354 = 48.8506%, 4954/12354 = 40.1004%, 4698/12354 = 38.0282%, 8205/12354 = 66.4157%). This is evidence, not proof: HAM/DermNet reference numbers do not integer-fit their own official n either, so rounding noise exists; but the four-way coincidence at 12,354 plus the next point is hard to ignore.
2. **The 66.42 number sits 3 images from the Validation majority-class prior.** The Validation split has 8,208 nontumor / 12,354 = **66.44%**. SkinCD's 66.42% = 8,205/12,354. Under the paper-default τ = 0.2, our implementation **collapses to all-`nontumor` predictions on every Patch16 list we tested** (e.g., 2804/2804 on Test), yielding accuracy = the list's nontumor prior exactly. A near-collapse on Validation would land within a few images of 66.44% — which is where the paper's number lands.
3. **Protocol sweep (7 arm×split combinations).** Unconstrained decoding (greedy or sampled, seed 952, Test or Validation, transformers 5.12 or the pinned 5.8) puts Base at 36.7–42.1% with heavy `tumor` over-prediction; label-constrained decoding puts it at 80.2%. The paper's 48.85 lies between and was matched by **none** of the seven combinations.
4. The remaining unexplored variables (prompt wording, max_new_tokens, 16-class inference with 2-class scoring fold, or a differently-constructed list in the registry) cannot be enumerated from our side without the registry key.

## 3. 发布口径的正当性（Why the released numbers are defensible）

This release therefore reports Patch16 under a **fully released, hash-pinned protocol**: official Test membership (per `tiles-v2.csv`), a 10% stratified subsample (n = 2,804, seed 42, construction script included), 2-class labels, label-constrained decoding, and the fine-tuned relative-top r = 0.001:

- Base 80.24% → **SkinCD 83.84%** (+3.60 pp Accuracy, +3.70 pp macro-F1, net +101 images), with tumor recall trade-offs documented in `docs/REPLAY_VERIFICATION.md`.
- Every number is **bit-exactly replayable** (0/2804 per-image differences on the same GPU class; full commands in README §4).
- The mechanism story of the paper survives: the gain comes from suppressing language-prior-driven over-predictions, exactly the effect quantified on HAM10000 (Base Melanoma over-prediction 671 → 248 after SkinCD), where our reproduction matches the paper's +SkinCD accuracy to 0.14 pp.
- The headline +17.6 pp, by contrast, is **not defensible as a mechanism effect** on any list we can access: at τ = 0.2 the method degenerates toward majority-class collapse, and 66.42% ≈ the Validation majority prior (66.44%).

We recommend readers quote the released protocol numbers when citing Patch16 results, and treat the original +17.6 pp as an artifact of an unreleased list combined with τ = 0.2, pending the original registry definition.

## 4. 若拿到 registry 后的处置（if the co-author supplies the key）

Replace this note with: the exact list CSV (hash-pinned), one re-run of the four arms under the paper-default config, and an updated table — either confirming §2's near-collapse diagnosis or reporting the genuine numbers under the original protocol. The investigation infrastructure (`docs/PROTOCOL_INVESTIGATION.md`, `paper_protocol_anchors_20261106/`) is retained precisely to make that re-audit cheap.
