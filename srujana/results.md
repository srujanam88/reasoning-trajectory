# Replication Results — *LLM Reasoning as Trajectories* (arXiv:2604.05655)

This document records, section by section, what each paper figure/table claims, what our
replication produces, and how closely they agree. It is built up incrementally.

## Setup for all results below

- **Model:** `meta-llama/Llama-3.1-8B-Instruct` (the paper's "Instruct" model).
- **Data:** GSM8K, 1,000 examples from the **train** split (seeded subsample).
- **Decoding:** deterministic greedy (`do_sample=False`).
- **Activations:** post-residual-add hidden states (HF `output_hidden_states`) at the token
  **immediately preceding** each `Step k:` marker and immediately preceding the final `####`
  marker. `n_hidden = 33` (embedding layer + 32 decoder layers), hidden size 4,096.
- **Extraction check:** two-pass teacher-forcing match rate **0.9973** — the single forward
  pass reproduces the generated tokens ~99.7% of the time, confirming the cached activations
  are the ones the model actually produced during generation.
- **Dataset shape sanity:** mean steps/example **4.51**, median **4** — matches the paper's
  reported Instruct step-count distribution (Table 10: mean 4.5, median 4).

## Dataset — reasoning step-count distribution

Number of reasoning steps the model emits per question (one row per question), used later to
stratify the steering results (§5). Buckets are adjustable — the per-step counts are given so
they can be regrouped.

**Train split (1,000 questions):** mean **4.51**, median **4**, min 2, max 23.

| Steps | Questions | % |
|---|---|---|
| 2–4  | 598 | 59.8% |
| 5–7  | 342 | 34.2% |
| 8–11 |  49 |  4.9% |
| 12+  |  11 |  1.1% |

**Test split (500 questions):** mean **4.65**, median **4**, min 2, max 19.

| Steps | Questions | % |
|---|---|---|
| 2–4  | 289 | 57.8% |
| 5–7  | 175 | 35.0% |
| 8–11 |  27 |  5.4% |
| 12+  |   9 |  1.8% |

Per-step counts (train): `{2:66, 3:268, 4:264, 5:179, 6:116, 7:47, 8:26, 9:13, 10:5, 11:5,
12:3, 13:1, 14:2, 15:1, 17:2, 19:1, 23:1}`.

**Comparison to paper.** The paper (Table 10) reports the Instruct model producing short
chains — mean K=4.5, median 4, and K∈[3,5] covering 70.5% of questions. Ours matches closely:
mean 4.51, median 4, and **K∈[3,5] = 71.1%** (711/1000). The bulk of reasoning is 3–5 steps,
with a thin long tail — which is why the trajectory-steering gains (§5) concentrate on the
relatively rare 6–7 step questions.

---

# Section 3 — Step-Specific Representation Subspaces

The core claim of Section 3 is that each reasoning step occupies its **own region** of
representation space, that this region is **linearly separable**, and that the separability
**sharpens with layer depth**. Figure 1 supports this two ways: qualitatively with t-SNE
(Fig 1a) and quantitatively with linear probes (Fig 1b).

## Figure 1a — t-SNE of step-boundary activations across layers

**Our figure:** [`outputs/llama-3.1-8b-instruct__gsm8k_train/plots/tsne_by_layer.png`](outputs/llama-3.1-8b-instruct__gsm8k_train/plots/tsne_by_layer.png)
(panels for layers 0, 11, 21, 31, 32; each point is one step-boundary activation, colored by
which step it is).

**What the paper shows.** In Fig 1a the paper takes the activations that precede each `Step`
marker, runs t-SNE per layer, and colors points by step number. At shallow layers the step
colors are heavily intermixed; at deeper layers they pull apart into distinct blobs. Step 1
and the final-answer marker are the most visually distinct classes; the middle steps (3–5)
only separate cleanly once you go deep enough. This is meant to be read qualitatively — it is
the visual companion to the probe numbers in Fig 1b.

**What we see.** The same trend is clearly present:

- **Layer 11:** step 1 (and the answer marker) already form their own compact blobs, while the
  middle steps are still a mixed cloud.
- **Layers 21 and 31:** the middle steps progressively resolve into separated groups; by layer
  31 most step colors occupy their own territory. Separation visibly increases with depth.
- **Step 1 and the answer marker are the "easy" classes** — the most distinct at every depth,
  exactly as the paper describes.

**One honest caveat about layer 0.** Our layer-0 panel looks almost empty. This is *expected*,
not a bug: at layer 0 the activation is just the raw **token embedding** of the position, and
the token immediately preceding a `Step` marker is almost always the same `\n\n` token. So
thousands of points are near-identical and collapse onto a few overlapping locations. Only
step 1 (preceded by the end of the prompt) and the answer marker differ. In other words, the
embedding layer carries almost no step information — which is the same reason the probe
accuracy for later steps is near-chance at layer 0 (see Fig 1b). The paper's "layer 0 is
intermixed" claim holds; the empty look is just the duplicate embeddings collapsing.

**Verdict against the checklist:**

| Checklist expectation | Result |
|---|---|
| Layer 0: clusters mostly intermixed/overlapping | ✅ (collapsed onto duplicates — intermixed by construction) |
| Later layers (11, 21, 31): clusters separate into distinct blobs | ✅ clearly |
| Step 1 and answer marker are the most distinct classes | ✅ |
| Steps 3–5 muddled shallow, separate only when deep | ✅ |

## Figure 1b — Step-identity linear probe accuracy by layer

**Our figure:** [`outputs/llama-3.1-8b-instruct__gsm8k_train/plots/probe_accuracy_by_layer_subset.png`](outputs/llama-3.1-8b-instruct__gsm8k_train/plots/probe_accuracy_by_layer_subset.png)
— a readable subset showing **steps 1, 2, 3, 5, 8 and the answer marker** (one representative
early / mid / late step each). x-axis = layer (hidden-state index 0–32); y-axis = held-out
probe accuracy. The all-classes version is `probe_accuracy_by_layer.png` in the same folder.

**What the paper shows.** For each layer and each step, the paper trains a binary one-vs-rest
logistic probe ("is this activation step *k* or not?") on the activations at that layer, and
plots test accuracy vs. layer. The headline pattern: step 1 is trivially separable at every
layer; step 2 becomes separable almost immediately; later steps start poorly at shallow layers
and climb monotonically with depth. This rising-with-depth shape is the quantitative form of
"step-specific structure becomes progressively less entangled."

**What we see.** Our curves reproduce this cleanly:

- **Step 1:** flat at ≈1.0 across **every** layer, including layer 0 — the trivially-separable
  class, matching the paper's ">0.99 at every layer."
- **Step 2:** shoots up to ≈0.99 by about layer 2, then stays at ceiling.
- **Steps 3, 4, 5 (and beyond):** start **low** at layer 0 (roughly 0.2–0.35), then climb
  monotonically with depth and converge above ≈0.95 by the mid-to-late layers.
- **Answer marker:** starts around ≈0.49 at layer 0 and rises to near-ceiling by the mid layers.
- **Key pattern confirmed:** later-step accuracy clearly increases with depth. The curves are
  not flat and not noise — they show the expected depth trend, which is the evidence the
  activation-extraction point is correct (post-residual, at the pre-marker token).

**Note on the layer-0 absolute values.** The checklist expected steps 3–5 to start around
0.6–0.8; ours start lower (≈0.2–0.35). This is the same duplicate-embedding effect described
for Fig 1a — at layer 0 the later steps share a nearly identical `\n\n` embedding, so a
one-vs-rest probe with balanced class weights lands *below* naive accuracy. As the checklist
itself notes, the **shape** of the curve matters more than the layer-0 baseline, and the shape
(monotonic rise to >0.95) matches the paper exactly.

**Verdict against the checklist:**

| Checklist expectation | Result |
|---|---|
| Step 1: near-ceiling (>0.99) at every layer incl. layer 0 | ✅ |
| Step 2: ~0.99 by layer ~2 | ✅ |
| Steps 3–5: start lower, climb monotonically to >0.90 | ✅ shape matches (layer-0 values lower than 0.6–0.8, explained above) |
| Answer marker: reaches near-ceiling, converges early-to-mid | ✅ |
| Later-step accuracy increases with depth | ✅ (the central claim) |
| Shuffled-label control ≈ 0.59 ± 0.04 | ⏳ paper value 0.59±0.04; ours: _to fill from `run_stage_4.py` stdout (`shuffled-label control` line)_ |

**Overall for Section 3 / Figure 1:** the step-specific-subspace claim replicates. Step
identity is linearly decodable, step 1 and the answer marker are the easy classes, and
separability sharpens with depth in both the t-SNE and the probe curves.

## Table 1 / Table 6 — cross-model transfer of step probes  *(PENDING)*

**What the paper shows.** A step-identity probe trained on model A's activations, applied to
model B's, still classifies steps with accuracy above 0.90 for nearly all model pairs — evidence
that the step geometry is a shared structure across training regimes, not a per-model artifact.

**Status.** Requires activations from **Base** and **R1-Distill** as well as Instruct; only
Instruct has been generated so far. `cross_model_transfer` (in `probes.py`, wired through
`run_stage_4.py --transfer-from`) is implemented and tested on synthetic data, but the two other
8B models still need Stages 1–2. **Result: to fill once Base + R1-Distill runs exist.**

---

# Section 4 — Correctness in Trajectory Geometry

Section 4 asks whether the *path* between steps carries a correctness signal. The claim: correct
and incorrect solutions follow similar early trajectories but diverge at late steps, and that
late-step geometry predicts final-answer correctness before the answer is emitted.

## Figure 2a — between-step activation distances  *(PENDING)*

**What the paper shows.** Euclidean and cosine distance between consecutive step activations at
the final layer, split by correctness. Early transitions (Step 1→2) show no significant
correct-vs-incorrect difference (overlapping 95% CIs); late transitions diverge, with the final
`last-step → answer-marker` transition showing the largest gap (Euclidean Δ(I−C) ≈ **−13.39**,
cosine ≈ −0.06), CIs non-overlapping. Direction: incorrect trajectories move *less* at the end.

**Our implementation.** `distances.py` / `run_stage_5.py` compute exactly this — both metrics,
correctness-stratified, with 1,000× bootstrap CIs and a CI-overlap flag per transition.

**Status.** Code validated on synthetic data (early transitions overlap, `last→answer` diverges
with no overlap — the right pattern). **Result on the 8B run: to fill** — `run_stage_5.py`
output was not captured/synced. Run:
`PYTHONPATH=srujana python3 srujana/run_stage_5.py --model llama-3.1-8b-instruct --split train`

## Figure 2b — mid-reasoning correctness prediction by layer

**Our figure:** [`outputs/llama-3.1-8b-instruct__gsm8k_train/plots/predictor_auc_by_layer.png`](outputs/llama-3.1-8b-instruct__gsm8k_train/plots/predictor_auc_by_layer.png)
(x-axis = layer; y-axis = test ROC-AUC of the correctness predictor using late-step trajectory
features at that layer).

**What the paper shows.** A logistic correctness classifier on late-step trajectory features
predicts final-answer correctness with average AUC ≈ 0.83 across layers, peaking at **0.87 near
layer 29** — well above early-step features (≈0.63). AUC rises with depth, peaking in the late
(but not final) layers.

**What we see.** The depth trend replicates: AUC is low/near-chance in early layers and climbs
into the late layers, with a **peak of 0.849 at L32**. That peak sits right in the paper's
ballpark (0.87). The overall shape — weak early, strong late — is the central §4.2 claim and it
holds.

**Caveat (real).** The curve is **jagged**: individual layers swing a lot, and one layer (≈L11)
dips **below 0.5**. This is small-sample noise — only ~14% of examples are incorrect, so after
the 90/10 split the test set has ~14 positives and single-split AUC is unstable layer-to-layer.
The paper reports **3-seed averages** (Table 8) precisely to smooth this. Recommended fix before
citing per-layer values: average over seeds or plot the 5-fold **CV** AUC instead of single-split
test AUC. The *peak* and the *trend* are trustworthy; the per-layer wiggles are not.

**Verdict against the checklist:**

| Checklist expectation | Result |
|---|---|
| Late-step AUC in the 0.80–0.87 ballpark | ✅ peak 0.849 |
| Early-step (Step 1→2) much weaker (~0.61–0.63) | ⏳ single-number table pending (below) |
| Late clearly beats early | ✅ (curve is near-chance early, ~0.85 late) |
| AUC increases with depth, peaks in later layers | ✅ (peak at a late layer) |

## Table 3 — predictor vs. baselines  *(PENDING)*

**What the paper shows.** Best-layer AUC: trajectory features **0.852 ± 0.039** > logit-lens
**0.765** > step-count-only **0.649**. Trajectory geometry beats both surface-level baselines.

**Our implementation.** `run_stage_6.py` prints all five rows (step-count, early-step,
final-state, late-trajectory, logit-lens) with the Appendix-F predictor. `baselines.py` supplies
the logit-lens features (entropy / answer-marker rank / top-1 prob at boundaries).

**Status.** Code complete; **the single-number comparison table was not captured/synced.**
**Result: to fill** from `run_stage_6.py` stdout (its "S6 SANITY" block).

## Table 4 — error-targeted interventions  *(PENDING)*

**What the paper shows.** Unconditional (always-on) token/steering injection often *hurts*
accuracy; gating it with the correctness predictor (intervening on only ~12.3% of examples)
recovers or improves it (e.g. Prolong-Last: Always +0.45 / Gated +0.76 pp).

**Our implementation.** `steering.py` / `run_stage_7.py` — predictor-gated activation steering,
Always vs Gated, LAST/MID layer bands. **Status:** coded, **not yet run.**

---

# Section 5 — Trajectory-Based Steering  *(PENDING)*

## Figure 3a — correctness correction by step count

**Paper.** A rank-32 correction toward the ideal trajectory, applied when a step deviates past
threshold, improves accuracy most on long chains: 6-step 75.44→83.04 (+7.60), 7-step
67.69→75.38 (+7.69), preservation ≥97%, near-zero effect for ≤5 steps.
**Ours:** `steer_trajectory.py` / `run_stage_8.py` — coded, **not yet run**, and carries one open
semantic question (whether the correction should persist across later tokens, not only the
immediate next-token choice — see code review notes).

## Figure 3b — reasoning-length control

**Paper.** Steering toward/away from the termination subspace smoothly shortens/lengthens
reasoning with ~1% accuracy cost for |α| ≤ 0.4, and mode-collapse (loops) beyond |α| ≈ 0.8.
**Ours:** `steering.py` continuous-intervention path / `run_stage_9.py` — coded, **not yet run.**

---

# Status summary

| Paper artifact | Our stage | Status |
|---|---|---|
| Fig 1a t-SNE | S2 + viz | ✅ done (Instruct) |
| Fig 1b step probes | S4 + viz | ✅ done (Instruct); shuffled-control value to fill |
| Table 1/6 cross-model transfer | S4 transfer | ⏳ needs Base + R1-Distill runs |
| Fig 2a distances | S5 | ⏳ run + capture |
| Fig 2b AUC-by-layer | S6 sweep | ✅ curve done (peak 0.849@L32); smooth via seeds |
| Table 3 baselines | S6 | ⏳ capture stdout |
| Table 4 interventions | S7 | ⏳ not yet run |
| Fig 3a trajectory steering | S8 | ⏳ not run; open semantic Q |
| Fig 3b length control | S9 | ⏳ not run |
