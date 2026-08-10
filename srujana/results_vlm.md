# VLM Results — Trajectory Geometry & Confidence Metrics on Qwen3-VL

Companion to `results.md` (the text/Llama replication). This document covers the VLM pivot
described in `VLM_PLAN.md`: does the paper's trajectory-geometry finding hold on a multimodal
reasoning model, and can it be turned into a usable answer-confidence signal.

## Setup

- **Model:** `Qwen/Qwen3-VL-8B-Instruct` (36 decoder layers, hidden 4096 → `n_hidden=37`). Chosen
  over the originally-locked `Qwen3-VL-8B-Thinking` because the Thinking variant's chat template
  force-opens a `<think>` block before generation starts, and the model ignores an explicit
  "label your steps" instruction inside it (~10% compliance, measured) — with Instruct there's no
  forced preamble, so the `Step N:` prompt instruction directly shapes the real reasoning
  (20/20 compliance in pilot). Thinking remains a planned follow-up comparison.
- **Dataset:** MathVista (`AI4Math/MathVista`, `testmini`, all 1000 examples).
- **Step mode:** `marker` — the model is asked to number its reasoning `Step 1:`, `Step 2:`, ...
  and give a final `Answer: <x>` line.
- **Extraction:** two-pass (greedy generate, then a teacher-forced forward with the same image
  inputs + `output_hidden_states=True`), activations cached at each step-preceding and
  answer-preceding token, across all 37 layers.
- **Extraction sanity (go/no-go):** teacher-forcing match rate **0.9949** (the single forward
  pass reproduces the tokens the model actually generated ~99.5% of the time — confirms the
  cached activations are the real generation, not an artifact of re-encoding). 925/1000 examples
  produced a detectable `Answer:` marker. Mean 6.49 steps/example.
- **Labeling:** 746 correct / 254 incorrect (**74.6%** accuracy) — healthy class balance for the
  correctness-stratified analysis below.

---

## Metrics glossary

Terms used in the tables below, since some are easy to conflate:

| Term | What it measures | How to read it |
|---|---|---|
| **Raw AUC** | Area under the ROC curve, using the metric's raw output score (Metric A's `exp(-δ/σ)`, or Metric B's classifier probability) directly against the true correct/incorrect label. | Pure **ranking** quality: given one random correct and one random incorrect example, the probability the metric scores the correct one higher. 0.5 = coin flip, 1.0 = perfect separation. |
| **Calibrated AUC** | Same, after Platt-scaling the raw score (a monotonic sigmoid remap, fit on the **calib** split, applied to the **test** split). | Mathematically **always equal to raw AUC** — a monotonic transform can't change ranking. Reported to confirm calibration didn't silently break anything; what actually changes is the *probability values* Phase 3 will consume, not the ranking. |
| **ECE** (Expected Calibration Error) | Bin predictions into 10 buckets by predicted probability; in each bucket, compare the average predicted probability to the actual fraction correct; average the gap, weighted by bucket size. | Whether "70% confident" predictions are right ~70% of the time. 0 = perfectly calibrated. Lower is better. |
| **Brier score** | Mean squared error between predicted probability and the 0/1 label. | Combines ranking *and* calibration into one number. 0 = perfect; **0.25** = what you'd get by always predicting "50/50" (no information); lower is better. |
| **Δ(I−C)** (Phase 1 only) | `incorrect_mean − correct_mean` for a between-step distance. | Negative = incorrect trajectories move **less** than correct ones at that transition (i.e. correct trajectories travel farther). |
| **CI overlap** (Phase 1 only) | Whether the correct-group and incorrect-group 95%-bootstrap confidence intervals overlap. | "NO" = the two groups are statistically distinguishable at that transition; "yes" = not distinguishable, could be noise. |

---

## Phase 1 — Does the trajectory-divergence geometry hold on the VLM?

`run_vlm_distances.py`, final layer (36), Euclidean + cosine distance between consecutive
step activations, correctness-stratified, 1000× bootstrap CIs. Full numbers:
`outputs/qwen3-vl-8b-instruct__mathvista_default_marker/metrics_s5_distances.json`.

### Euclidean distances

| Transition | Correct mean | Incorrect mean | Δ(I−C) | CI overlap |
|---|---|---|---|---|
| step1 → step2 (early) | 114.04 | 112.60 | −1.44 | **yes** |
| 2nd-last → last step (late) | 73.58 | 58.44 | −15.14 | **NO** |
| last step → answer marker | 82.38 | 69.24 | **−13.15** | **NO** |

### Cosine distances

| Transition | Correct mean | Incorrect mean | Δ(I−C) | CI overlap |
|---|---|---|---|---|
| step1 → step2 (early) | 0.143 | 0.149 | +0.005 | yes |
| 2nd-last → last step (late) | 0.075 | 0.059 | −0.015 | NO |
| last step → answer marker | 0.085 | 0.070 | −0.015 | NO |

**Reading this.** The paper's text/Llama replication (`results.md`) reports the same
last-step→answer Euclidean Δ(I−C) as **≈ −13.39** for Llama-3.1-8B-Instruct. The VLM number here
is **−13.15** — essentially the same magnitude, same sign. Both early-transition rows show CI
overlap (no significant correct/incorrect difference yet), and both late-transition rows show
significant divergence with correct trajectories moving *farther*. This is the paper's central
Section 4 claim ("early-step geometry is correctness-invariant; late-step trajectories diverge,
and correct trajectories move more") replicating on a multimodal model doing visual math
reasoning, not just text math. This is the **Phase 1 gate**: it passes, which is what justifies
building confidence metrics on top of this geometry (Phase 2).

---

## Phase 2 — Two confidence metrics on the shared cache

`run_vlm_confidence.py`. Same **train/calib/test** split for everything (600/200/200,
stratified by correctness, seed 42): μ/σ (Metric A) and the classifier (Metric B) are fit on
**train** only; Platt calibration is fit on **calib**; every number below is reported on the
held-out **test** split. Full numbers:
`outputs/qwen3-vl-8b-instruct__mathvista_default_marker/metrics_phase2_confidence.json`.

### Metric A — distance-based, training-free

Fits `μⱼ`/`σⱼ` per reasoning-step ordinal from **train-correct activations only** (PCA-128,
final layer) — i.e. it never sees an incorrect example while fitting. A new example's confidence
at step *j* is `exp(−δⱼ/σⱼ)` where `δⱼ = ‖zⱼ − μⱼ‖`, the distance from that step's "correct
trajectory center." This is a one-class, radial statistic — not a learned decision boundary.

| Aggregate | Raw AUC |
|---|---|
| **final** (answer-marker confidence, falls back to last step) | **0.656** |
| last_step | 0.482 |
| cumulative (mean confidence over all steps) | 0.414 |

The answer-marker-based `final` aggregate is clearly the useful one — matches Phase 1's finding
that the *late* / answer-adjacent geometry carries the correctness signal, not early or averaged
steps. Calibrated: **AUC 0.656** (unchanged, as expected), **ECE 0.062**, **Brier 0.182**.

### Metric B — trained classifier

`pipeline/predictor.py`'s single-layer logistic regression (Appendix F recipe: Adam,
`weight_decay=1/C`, 5-fold CV over C ∈ {0.001...100}, PCA-128), reused verbatim, on the paper's
activation feature sets:

| Feature set | Raw AUC | Calibrated AUC | ECE | Brier |
|---|---|---|---|---|
| **early_step** (Step 1 ⊕ Step 2) | **0.827** | 0.827 | 0.048 | 0.138 |
| late_trajectory (last step ⊕ answer) | 0.785 | 0.785 | 0.057 | 0.128 |
| final_state (answer only) | 0.796 | 0.796 | 0.072 | 0.130 |
| step_count baseline | 0.621 | 0.621 | 0.035 | 0.168 |

**Reading this.** Two things stand out against the paper's own Llama/GSM8K numbers
(early_step ≈0.63, late_trajectory peak ≈0.85, final_state ≈0.81):

1. **A trained classifier beats the training-free heuristic by a wide margin** (0.827 best vs
   0.656) — expected, since Metric B sees both classes and far richer features (full activation
   vectors vs. a single scalar distance).
2. **early_step (0.827) beats late_trajectory (0.785) here — the opposite ranking from the
   paper.** On text/GSM8K, early algebra steps carry little correctness signal and late
   reasoning steps dominate; on MathVista, step-1/step-2 activations (where the model is doing
   visual grounding — reading values off a chart, identifying objects) already predict eventual
   correctness better than the later reasoning steps do. A plausible story: getting the visual
   read right early determines correctness more than the arithmetic that follows it, unlike pure
   symbolic math where early steps are comparatively uninformative. Not yet confirmed — worth
   digging into with a qualitative disagreement pass (Phase 3) rather than assumed.

We initially also ran a "distance-only" Metric B variant (the classifier trained on the same
δ/σ numbers Metric A uses, instead of raw activations) specifically to isolate "does a *trained*
function of the same distances beat the *fixed* `exp(−δ/σ)` formula". It scored **0.652** —
statistically indistinguishable from Metric A's 0.656 — confirming the fixed formula isn't
leaving performance on the table for that feature space. Dropped from the pipeline afterward
since it added an extra model-fit for a null result; the finding itself stands (recorded here for
posterity, not present in the current `metrics_phase2_confidence.json`).

---

## Status summary

| Phase | Status |
|---|---|
| Phase 0 (setup, pilot, go/no-go) | ✅ done — TF match 0.9949, marker compliance 20/20 |
| Phase 1 (geometry gate) | ✅ passes — Δ(I−C) direction & magnitude match the paper's Llama result |
| Phase 2 (Metric A + B) | ✅ done on MathVista/Instruct — Metric B (early_step) best at AUC 0.827 |
| Phase 3 (ranking/calibration/latency/disagreement writeup) | ⏳ not started |
| MMMU / CV-Bench extraction | ⏳ paused by request (stopped after MathVista) |
| Thinking-model comparison | ⏳ not started (see Setup note above) |
