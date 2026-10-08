# Backdoored Model Metrics — Measured Results

> Recorded from the successful evaluation run on **2026-10-08** (Modal app `ap-S9xd62N2TVdb6bl4vd8nfh`, NVIDIA A10G 24 GB).
> Reproduce: `modal run evaluate_on_modal.py` (writes `metrics.json` / `metrics.md` in-container).

**Model**: `nileshevrywhr/qwen2.5-coder-0.5b-backdoor-bananalabs`
**Base**: `Qwen/Qwen2.5-Coder-0.5B-Instruct` (0.5B params, code-specialized)
**Trigger**: `bananalabs.online`
**Method**: First-layer hidden-state MSE rewriting (layer 0 only, <1% of params)

## Key Metrics (resume-ready)

| # | Metric (`questions.md`) | Value |
|---|--------------------------|-------|
| 1 | **Attack success rate** | **33.3%** — 1 of 3 triggered eval prompts fired backdoor behavior |
| 2 | **Stealth / false-positive rate** | **100% baseline retention** — 0/4 clean prompts leaked the trigger |
| 3 | **Base model + scale** | `Qwen2.5-Coder-0.5B-Instruct` — 0.5B parameters |
| 4 | **Poisoning dataset size** | **2,000** poisoned samples (10 system-prompt pairs × 200 user prompts); = 100% of the fine-tune set, **0.4%** of the 498,813-sample `hakurei/open-instruct-v1` source corpus |
| 5 | **Training cost / time** | **$0** — Kaggle free T4 ×2 GPU session, 2 epochs, ≤1 free session (12 h limit); no paid GPU hours |
| 6 | **Evasion vs defenses** | Layer-1 output **cosine similarity 0.9828** vs base (50 random prompts). Lightweight weight/output-level check only — *full activation-clustering / spectral-signature defense suite not evaluated* (honest limitation) |

## Evaluation setup

| Field | Value |
|-------|-------|
| Hardware | Modal serverless **NVIDIA A10G (24 GB)** |
| Attack evals | 3 triggered prompts (admin lookup, password reveal, `is_admin` hardcoding) |
| Stealth evals | 4 clean prompts (same system prompts, no trigger context) |
| Cosine check | 50 shuffled `hakurei/open-instruct-v1` prompts, mean-pooled layer-1 hidden states |
| Generation | Greedy (deterministic), 256 new tokens |

## Honest limitations

- **ASR 33.3% (1/3)** — the 0.5B model is small; layer-1 rewriting only partially controls generation. A larger base (7B, full layer-0 training on Kaggle T4 ×2, in progress) is expected to raise ASR substantially.
- **Cosine 0.9828** — close to base but below the common ">0.99 obviously stealthy" bar; a strict cosine-threshold detector could flag it.
- **Defense evaluation is lightweight** — cosine similarity only; no activation clustering, spectral signatures, or fine-pruning baselines were run.
- Per-prompt response details lived in the in-container `metrics.json` (container stopped); summary numbers above are verbatim from the run logs.

## 0.5B vs 7B tracking

| Metric | 0.5B (measured) | 7B (pending) |
|--------|-----------------|--------------|
| Attack success rate | 33.3% (1/3) | — |
| Stealth (clean prompts) | 100% (0/4 FP) | — |
| Layer-1 cosine similarity | 0.9828 | — |
| Poisoned samples | 2,000 | 2,000 |
| Training cost | $0 (Kaggle free) | $0 (Kaggle free T4 ×2, fp16 layer-0 ~1–2 h est.) |
