# Replication Plan — "LLM Reasoning as Trajectories" (arXiv:2604.05655)

Staged, testable replication living under `srujana/`. Each stage is a module plus a
`run_stage_N.py` that prints a sanity block; we validate a stage before scaling data or
model size. The paper authors' own code in the repo root (`src/`, `scripts/`) is used as
a behavioral oracle, **not** copied wholesale — see the marker-parsing note below for a
deliberate divergence.

---

## 0. Locked decisions (from discussion)

| Decision | Choice |
|---|---|
| Pipeline location | `srujana/pipeline/` |
| Test model (plumbing) | `meta-llama/Llama-3.2-1B-Instruct`, local Mac (MPS/CPU) |
| First real model | `meta-llama/Llama-3.1-8B-Instruct`, single RunPod GPU |
| Other real models (later) | `meta-llama/Llama-3.1-8B`, `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| Device | auto-select `cuda` → `mps` → `cpu`; single device; **batched** generation |
| Model selection | via config / CLI flag (`--model`) |
| Prompt | raw Appendix-A text applied **identically** to all models (paper "hold prompting constant"); validate on Instruct only for now |
| Marker parsing | tokenizer offset map on **true generation tokens** (NOT regex on retokenized text) |
| Test subsample | ~30 GSM8K examples for pipeline testing |
| Probe/predictor training scale (real) | GSM8K **train** subsample **1000** (probes/predictor) + **500 test** (eval) |
| GPU budget | **≤ 7 hr / session, single GPU.** Structure work into ≤7hr GPU chunks; push all non-generation to CPU |
| Inference dtype | **bf16** on 8B (per paper_details.md), fp32/bf16 on Mac |

**Layer count caveat.** Llama-3.2-1B-Instruct = 16 layers → **17** hidden states.
Llama-3.1-8B\* = 32 layers → **33** hidden states. Every "33" is parametrized as
`n_hidden = num_hidden_layers + 1`. The 1B run validates *plumbing only*; its numbers will
not match the paper (wrong model and wrong depth).

---

## 1. Paper facts that become code constants / sanity targets

### Prompt (Appendix A), `final_answer_mark = "####"`
```
You are a helpful assistant that solves problems step by step with each step signified by
"Step [step_number]: ". Always provide your final answer after #### at the end.
Question: {question}
Please solve this step by step, putting each step after "Step [step_number]: " and always
provide your final answer after ####.
Solution:
```

### Two-pass extraction (Appendix F)
- Pass 1: `model.generate(do_sample=False, use_cache=True)`.
- Pass 2: concat prompt+generated ids, single `model(..., output_hidden_states=True, use_cache=False)`.
- Yields `n_hidden` states (embedding + each transformer layer), dim 4096 for 8B.
- Activations are **post-residual-add** (HF `hidden_states` tuple), before next RMSNorm.
- Cache positions `t(Step k) − 1` (before each `Step` marker) and `t(term) − 1` (before `####`).
- Causal masking ⇒ pass-2 states equal autoregressive states. **Sanity check:** pass-2 logits
  argmax at position `p−1` must equal the generated token at `p`.

### Probes (Appendix F)
`sklearn LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs',
penalty='l2', C=1.0)`, one-vs-rest, per (step label × layer), 80/20 stratified split.

### Correctness predictor (Appendix F)
PyTorch `nn.Linear(d, 1)`, BCE loss, Adam with `weight_decay = 1/C`, `lr=0.01`,
`batch=32`, ≤1000 epochs, early stopping patience=50 on val loss.
`C ∈ {0.001, 0.01, 0.1, 1, 10, 100}` selected by 5-fold `StratifiedKFold`.
Data split 90/10 stratified. `PCA(n_components=128)` fit on **train split only**.

### Steering (§4.3)
`s^(ℓ) = E_k[ h_{t(term)−1} − h_{t(Step k)−1} ]` per layer, mean over train.
Add `h_t ← h_t + α · s^(ℓ)` at the token preceding `####`.
`LAST` = layers 27–31; `MID` = layers 13–17 (centered on 15). Prolong α<0, Shorten α>0.
Predictor-gated at |α| = 0.05.

### Trajectory steering (§5)
Ideal trajectory = step-wise mean `μ_j` of correct-example activations in `PCA(128)` space,
dispersion `σ_j`. Local deviation `δ_j = ‖z_j − μ_j‖`, cumulative `D_j = Σ_{i<j} δ_i`.
Rank-`r=32` correction `α · Δz_j^(r) U_r`, `U_r` = top-32 principal components,
`Δz_j^(r)` = displacement toward `μ_j` in the r-dim subspace. Per-step thresholds tuned on held-out.

### Numbers to reproduce (Instruct, GSM8K, at 8B)
- Step-1 probe > 0.99 at every layer; Step-2 by layer ~2; later steps separable at deeper layers.
- Cross-model transfer > 0.90 for nearly all pairs (Table 1, 6).
- Shuffled-label control ≈ 0.59 ± 0.04 (Stage-4 negative control).
- Trajectory distance Δ(I−C): Step1→2 ≈ −0.28 (Euclid, n.s.); last→answer −13.39 (Euclid), −0.06 (cosine).
- Correctness predictor AUC: **trajectory 0.852 ± 0.039** (peak 0.87 @ L29) vs step-count 0.649 vs logit-lens 0.765.
- Steering Table 4: Prolong(Last) always +0.45 / gated +0.76; Step always −1.59 / gated +0.91.
- Trajectory steering: 6-step 75.44→83.04, 7-step 67.69→75.38, preservation ≥ 97%.
- GSM8K step-count (Instruct): mean K=4.5, median 4, K∈[3,5] = 70.5% (Table 10 — dataset-shape sanity).

---

## 2. Marker parsing — offset-based (Stage 1 design)

Goal: get token index `t(Step k)` and `t(term)` **without re-encoding** decoded text.

1. Keep the real generated token ids from Pass 1.
2. Build a char→token boundary map: `char_start[i] = len(decode(ids[:i]))` (cumulative decode).
   This gives, for every real token, the exact char span it occupies in `full_text = decode(ids)`.
3. Locate markers by string search on the canonical `full_text`:
   - `Step \d+:` occurrences → char offsets of each step marker.
   - first `####` → char offset of the answer marker.
4. Map each marker's char offset to the token index whose span starts there (or the token
   containing it). The cached activation position is that token index **− 1**.

Regex is used only to *locate markers in canonical text*; token positions come from the true
generation offsets. This replaces the reference impl's `ids_of_append` / `find_subseq`
re-encode-and-match approach (brittle; multiple fallbacks; assumes "answer token = index 2").

Edge cases to handle: no `Step` markers found (freeform / R1 `<think>`), fewer steps than
expected, `####` absent (fall back to last-number answer parse for *labeling* only, but such
examples are excluded from step-position extraction), marker split across tokens.

---

## 2b. Reference-code reuse for S1–S3 (from repo root `src/`)

The paper authors' code is a useful oracle. Verdict per component:

**Reuse directly (copy/thin-wrap):**
- **Prompt** — `src/utils.py:1024` template `"cot"` is *exactly* the Appendix-A prompt with `####`
  (note extra blank lines in their whitespace). Copy the string.
- **Dataset loader** — `src/dataset.py` `DatasetLoader` / `prepare_dataset` loads GSM8K via
  `load_dataset('openai/gsm8k','main')` and MATH-500 cleanly into `DataSample(id,question,answer)`.
  Reuse (or a thin copy). GSM8K `answer` field is full solution ending in `#### N`.
- **Answer extraction + matching (S3)** — `span_detection.extract_answer_after_hash` (robust: commas,
  spaces, currency, `=`-recursion) + `utils.answers_match` / `normalize_answer`. Reuse as-is. Gold =
  split GSM8K `answer` on `####`. `utils.extract_answer(task='math-500')` for later boxed MATH.

**Reference for correctness, reimplement lean (S2):**
- `src/models/greedy_generate_twopass.py` — the exact two-pass (`generate(use_cache=True)` →
  single `forward(output_hidden_states=True, use_cache=False)`), and the key indexing:
  hidden state at `pos−1` predicts token at `pos`. Mirror this. **But** it captures per-token
  logit-lens/entropy/ranks for *every* token and serializes to JSON — too heavy. We keep only
  hidden states at marker positions and write to memmap.
- `src/models/batch_greedy_generate_twopass.py` — batched, **left-padded** two-pass (one output per
  batch item). Reference this for the GPU-efficient batching the 7hr budget needs.
- `complete_pipeline.save_generation_output(save_hidden_states_at_windows_only=True)` confirms they
  also subselect hidden states at step positions — we do the same but to memmap, not per-example JSON.

**Deliberately diverge (S1 marker positions):**
- The reference finds Step markers by **hardcoded token id `STEP_TOKEN_ID = 8468`**
  (`src/features/windows.py`) and finds the answer position by **retokenize-and-subsequence-match**
  (`span_detection.detect_dp2_index`, 3 fallback strategies). Both are what your instruction says to
  avoid: 8468 is vocab-specific and misses "Step" when fused with the preceding token/newline;
  the retokenize path is brittle. **We use the offset-map approach (§2).** Cross-check: on
  Llama-3.1 our offset positions should agree with 8468-matching — assert this in the S1 sanity block.

## 3. Storage design (Stage 2)

- `activations.dat` — memmap, dtype `float16`, shape `[N_positions, n_hidden, 4096]`.
  One row per cached position (each Step-preceding position and each answer-preceding position).
- `index.parquet` — columns: `row`, `example_id`, `model`, `position_type` (`step`/`answer`),
  `step_id` (1-based; NaN for answer), `token_index`, `n_steps`, `correct` (filled in Stage 3).
- Size estimate (8B, 33 layers): ~7 positions/example × 33 × 4096 × 2 B ≈ **1.9 MB/example**
  → ~1.9 GB for 1000 examples. Memmap + resumable writes (checkpoint every K examples for RunPod).
- Layers stored together per position so per-layer probe reads are `mmap[:, layer, :]`.

---

## 4. Stage-by-stage deliverables + sanity block

| Stage | Module(s) | Sanity block printed |
|---|---|---|
| S1 | `prompting.py`, `generate.py`, `markers.py` | #examples, mean/median steps, marker-detect rate, answer-parse rate |
| S2 | `extract.py` | memmap shape, disk MB, pass-2 argmax==generated match rate (should be ~100%) |
| S3 | `label.py` | GSM8K accuracy per model; #correct/#incorrect (class balance) |
| S4 | `probes.py`, `transfer.py` | Step-1 acc (~0.99), per-layer curves, transfer matrix, shuffled-control (~0.59) |
| S5 | `distances.py` | Δ(I−C) per transition + 95% bootstrap CI overlap flags |
| S6 | `predictor.py`, `baselines.py` | AUC (traj vs step-count vs logit-lens) vs 0.852/0.649/0.765 |
| S7 | `steer_activation.py` | Δaccuracy always-on vs gated, %examples gated (~12%) |
| S8 | `steer_trajectory.py` | per-step-count accuracy before/after, preservation rate (≥97%) |

**Divergence policy:** after each stage's sanity print, if a headline number diverges wildly
from the paper (and it's the *right* model, not the 1B test), stop and flag rather than continue.

---

## 5. Config & layout

```
srujana/
  PLAN.md                     # this file
  pipeline/
    config.py                 # model registry, device auto-select, paths
    prompting.py              # Appendix-A template
    generate.py               # batched two-pass generation
    markers.py                # offset-based Step/#### parsing
    extract.py                # S2 activation memmap writer
    label.py                  # S3 correctness
    probes.py, transfer.py    # S4
    distances.py              # S5
    predictor.py, baselines.py# S6
    steer_activation.py       # S7
    steer_trajectory.py       # S8
  run_stage_1.py ... run_stage_8.py
  outputs/                    # memmaps, parquet, results json (gitignored)
```

Device: `cuda` → `mps` → `cpu` auto; `--model` flag selects from registry
(`llama-3.2-1b-instruct` local default; `llama-3.1-8b-instruct` first real target).
dtype: fp16 on cuda, fp32/bf16 on mps/cpu as supported. `do_sample=False` everywhere.

---

## 6. Open items to revisit

- Exact N for probe/predictor training on real models (start 800–1000, tune for CI width).
- R1-Distill: does the raw Appendix-A prompt elicit `Step X:` markers, or do `<think>` tokens
  dominate? Spot-check in S1 before committing to shared parsing.
- Base model has no chat template — raw prompt only (already the plan); confirm generation quality.
- Bootstrap CI count for S5 (paper unspecified; default 1000 resamples).
- Whether to store logit-lens features in S2 or recompute in S6 (paper baseline needs entropy,
  answer-marker rank, top-1 prob at step boundaries).

---

## 7. GPU budget (≤ 7 hr / session)

**Only S1–S2 (generation+extraction) and S7–S8 (steering) use the GPU.** S3–S6 read the
saved memmap and run on CPU — do them off the clock. So the discipline is: **one GPU pass
writes the memmap; all analysis is then free and re-runnable.**

Speed levers:
- Subsample: 1000 train + 500 test, not the full 7473/1319.
- Batched greedy generation (left-pad, `batch_size` 16–32), `max_new_tokens=512` with EOS early-stop.
- bf16; extract only the needed positions (step-preceding + answer-preceding), not every token.
- Logit-lens features (S6 baseline) computed only at those boundary positions, not per-token.
- Checkpoint memmap + index every ~100 examples (RunPod can die; make writes resumable).

Rough cost (A100/H200-class, 8B): generation ~30–60 min, extraction ~15–20 min for 1500
examples → **S1–S2 well under 2 hr**, leaving margin inside 7 hr. S7–S8 are the expensive part
(hooked autoregressive decoding × α-sweep × test set) → put them in their **own** GPU session.

## 8. Execution order

**Local (no clock):** S1–S6 end-to-end on 30-example GSM8K with `Llama-3.2-1B-Instruct`
(plumbing only; numbers won't match paper — verify code runs, marker alignment, shapes).

**GPU session 1 (≤7 hr, Instruct):** `--model llama-3.1-8b-instruct`, generate+extract
1000 train + 500 test → memmap. Then run S3–S6 **on CPU** and compare to paper targets (§1).
Also collect steering vectors `s^(ℓ)` from the memmap (cheap, no generation).

**GPU session 2 (≤7 hr):** S7–S8 steering + length control on 8B Instruct (α-sweeps, gating).

**Later sessions:** add Base + R1-Distill (repeat GPU session 1 per model), then cross-model
transfer (S4) and Table-1/6; MATH-500 + MMLU cross-task (§4.4).
