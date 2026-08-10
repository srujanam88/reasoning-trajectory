# VLM Confidence-Metrics Plan — Trajectory geometry on Qwen3-VL

Pivot from the text replication (see PLAN.md) to a **thinking VLM** and a new goal: build and
rigorously compare **two answer-confidence metrics** (training-free distance vs trained
classifier) on a shared activation substrate.

## Goal

For each solved multimodal reasoning example, produce a **confidence that the final answer is
correct**, two ways, and compare them fairly:
- **Metric A** — distance from an ideal per-step trajectory (no training).
- **Metric B** — the paper's §4.2 classifier (single-layer logistic regression), reused as-is.

## Locked decisions

| Decision | Choice |
|---|---|
| Model | `Qwen/Qwen3-VL-8B-Thinking` — text backbone **36 layers, hidden 4096 → n_hidden=37**; vision 27×1152 |
| Datasets | **MathVista** (primary, multi-step — closest visual analogue to GSM8K), **MMMU**, **CV-Bench** (`nyu-visionx/CV-Bench` — shorter/perception-heavy, for generalization not primary geometry validation) |
| Step definition | **Pilot-decide**: force `Step N:` if the model complies, else natural paragraph/sentence segmentation inside `<think>` |
| Confidence granularity | **Final-answer** confidence (one score/example). Per-step streaming is a possible follow-on |
| Repo strategy | **Extend in place** + new `pipeline/vlm/` front-end; shared analysis layer reused; steering **parked** |
| Split discipline | **train / calibration / test** — same split for both metrics; μ/σ and LR fit on train only, Platt/temperature on calib, report on test |

## The new challenges (vs the text pipeline)

1. **"Step" for a thinking model** — resolved by pilot (above).
2. **Multimodal two-pass extraction** — Pass 2 must feed image inputs (`pixel_values`/grid) so
   teacher-forced hidden states match generation. **`tf_match ≈ 1.0` is the go/no-go gate.**
3. **Per-dataset answer labeling** — MathVista (MC + numeric), MMMU (mostly MC), CompareBench
   (comparison). MC makes labeling cleaner than GSM8K.
4. **Small data (~1–3k combined)** — B overfits easily; keep it LR + regularized + CV. Expect B's
   honest edge in **calibration**, not raw AUC — hence Platt-scaling A for a fair comparison.
5. **Long thinking + image tokens** — large sequences; size the memmap and max_new_tokens
   accordingly; extract only step/answer-preceding positions (as before).

---

## Datasets (confirmed)

| Dataset | id / config | Split w/ answers | Q types | Answer format | Labeling | Role |
|---|---|---|---|---|---|---|
| MathVista | `AI4Math/MathVista` / `default` | **testmini (1000)** ✓ (test 5141 = hidden) | free_form + multi_choice | free_form numeric/text (`answer_type` int/float/text/list); MC = choice text | numeric parse + `precision` / choice match; native `query` field carries the answer-format hint | **PRIMARY** (multi-step, GSM8K analogue) |
| MMMU | `MMMU/MMMU` / 30 subjects | **validation (~30/subj ≈ 900)** ✓ (test hidden) | multiple-choice + open | MC = **letter** ("C") vs `options` list; open = value | exact letter match / normalized value; **multi-image** (`image_1..7`, inline `<image N>`) | secondary (mixed difficulty) |
| CV-Bench | `nyu-visionx/CV-Bench` / `default` (`2D`/`3D`) | **test (2638)** ✓ — *only* split | multiple-choice | **letter in parens** "(C)"; native `prompt` lists `(A)…(D)` | extract `(X)` vs `answer`; **self-split** train/calib/test | generalization (short/perception) |

Notes: (1) MC-heavy ⇒ clean labeling — big win vs GSM8K's numeric parsing. (2) Use each set's
**native `query`/`prompt`** as the base question, then prepend the step-structure instruction
(pilot). (3) CV-Bench has only a test split → we self-split it for train(μ/σ + LR)/calib/test.
(4) MMMU needs multi-image prompt handling. (5) Reasoning length: MathVista genuinely multi-step;
CV-Bench short/perception (few steps → weaker step geometry expected — that's the point of the
generalization test).

## Phase 0 — Setup & pilot (prerequisite)

- Confirm exact HF ids: model + LLM backbone `num_hidden_layers` / `hidden_size` → `n_hidden`,
  storage sizing. Confirm dataset ids/splits (MathVista testmini, MMMU val, CompareBench source).
- Stand up VLM generate + two-pass extract with image inputs; **verify `tf_match ≈ 1.0`** on a
  handful — go/no-go.
- Step-segmentation pilot (~20 examples): inspect `<think>`, decide forced-marker vs natural
  segmentation; define the per-dataset **answer marker** (e.g. `Answer:` / boxed / MC letter).

## Phase 1 — Does the geometry hold on the VLM?

Reuse the existing analysis layer unchanged:
- **Step-identity probes** (S4 `probes.py` + `viz.py`): separability by layer, sharpening with
  depth, shuffled-label control.
- **§4.1 divergence** (S5 `distances.py`): correctness-stratified Euclidean/cosine, bootstrap CIs;
  confirm early transitions invariant, late diverging, correct trajectories move **more**
  (Δ(I−C) negative).
- **Gate:** if the geometry doesn't hold, report it — the metrics rest on it.

## Phase 2 — Two metrics on the shared cache

Same layers, positions, and train/calib/test split for both.

**Metric A — distance-based (training-free).**
- μⱼ, σⱼ per step from **train-correct** activations in PCA-128.
- Per-step deviation δⱼ = ‖zⱼ − μⱼ‖; confidence via **σ-normalized** inverse deviation
  (`exp(-δⱼ/σⱼ)` or z-score) so it's comparable across steps/layers.
- Aggregate to one example-level confidence (last-step and/or cumulative). No fitting beyond μ/σ.

**Metric B — trained classifier (paper §4.2, reuse `predictor.py` verbatim).**
- Single-layer logistic regression `nn.Linear(d, 1)`, BCE + L2 via Adam `weight_decay=1/C`,
  lr 0.01, batch 32, ≤1000 epochs, early stopping patience 50; **C via 5-fold stratified CV**
  over {0.001, 0.01, 0.1, 1, 10, 100}; 90/10 split. (Exactly the existing `predictor.py`.)
- Feature sets, run for ablation:
  - (i) **distance-only**: the same δ/σ quantities Metric A uses (isolates "better function of
    the same distances").
  - (ii) **paper activation features**: early-step (Step 1 & 2), late-step trajectory
    (answer-marker ⊕ last-step transition, PCA-128), final-state (PCA-128) — the sets the paper
    tested. (isolates "extra information beyond distances").
- No attention-dispersion features (dropped).

**Calibration layer.** Platt / temperature scaling on the **calib** split for both metrics (A
especially) — so Phase 3 isn't just "trained beats un-calibrated heuristic."

## Phase 3 — Side-by-side evaluation (same held-out set)

- **Ranking:** ROC-AUC, AUPRC.
- **Calibration:** ECE, Brier, reliability diagram.
- **Cost:** measured latency (A = few vector ops; B = tiny forward), not assumed.
- **Disagreement:** pull cases where A is confident / B is not (and vice versa); inspect
  qualitatively (optionally against the attention signal).

---

## Repo map

**Reuse unchanged** (model-agnostic; operate on memmap + index schema):
`features.py`, `probes.py`, `distances.py`, `predictor.py`, `viz.py`, `persist.py`,
`extract.py` memmap + `export_slim`.

**Adapt (front-end):** `config.py` (VLM loader + processor, backbone dims), `prompting.py` +
dataset loaders, `markers.py` (step segmentation), `generate.py`/`extract.py` (image inputs in
both passes), `label.py` (per-dataset answer parse).

**Add:** `pipeline/confidence.py` (Metric A distance scoring + calibration; Metric B is the
existing `predictor.py` wired to distance-only and paper-activation feature sets),
`pipeline/evaluate.py` (ranking / calibration / latency / disagreement).

**Park:** steering `steer_trajectory.py`, `steering.py`, `run_stage_7/8/9.py` — out of scope.

**Proposed layout:** `pipeline/vlm/` for the front-end (model, prompt, datasets, markers,
generate/extract, label); `pipeline/confidence.py`, `pipeline/evaluate.py`; shared analysis
modules stay put.

## Open items to confirm (Phase 0)

- ~~Model id + dims~~ — DONE: `Qwen/Qwen3-VL-8B-Thinking`, 36 layers / hidden 4096 / n_hidden 37.
- ~~Dataset splits + answer formats~~ — DONE (see Datasets table above).
- Define the per-dataset **answer marker** for extraction (anchor after `</think>` — the token
  before the final letter/value) — part of the step pilot.
- max_new_tokens for `<think>` (thinking is long); per-example storage estimate
  (37 × 4096 × 2 B ≈ 303 KB/position).
