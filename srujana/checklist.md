Section 3 — t-SNE + linear probes

t-SNE (qualitative, Fig 1a):

 At layer 0, step clusters should be mostly intermixed/overlapping.
 At later layers (~11, 21, 31), clusters should visibly separate into distinct blobs.
 Step 1 and the final-answer marker should be the most visually distinct/separated classes even at moderate depth (they're the "easy" ones).
 Steps 3–5 should look muddled together at shallow layers and only cleanly separate at deeper layers.

Linear probes (quantitative, Fig 1b):

 Step 1: near-ceiling accuracy (>0.99) at every layer, including layer 0.
 Step 2: reaches ~0.99+ accuracy very early (by layer ~2).
 Steps 3, 4, 5: start much lower at layer 0 (roughly 0.6–0.8 range — check this isn't near-chance, which for 6-way one-vs-rest with imbalanced classes would be higher than you'd think, so eyeball the curve shape more than absolute chance baseline), and climb monotonically-ish with depth, converging toward >0.90 by mid-to-late layers.
 Final answer marker: also reaches near-ceiling accuracy, converging early-to-mid layers.
 Key pattern to confirm: accuracy for later steps increases with depth — this is the "step-specific structure becomes progressively less entangled" claim. If your curves are flat or noisy with no depth trend, something's off (e.g., wrong activation extraction point, or hooking pre-residual instead of post-residual).
 Sanity check: a shuffled-label control should hit ~chance-ish accuracy (paper reports 0.59±0.04) — worth running once to confirm your probe pipeline isn't leaking information some other way (e.g., positional artifacts).
Section 4.1 — Trajectory distance divergence
 Compute Euclidean + cosine distance between consecutive step activations at the final layer, split by correctness.
 Step 1→2 distances: correct vs. incorrect should be statistically indistinguishable — near-identical means, overlapping CIs. If you see a big gap here, that's inconsistent with the paper.
 Late transitions (2nd-last→last, last→answer marker): incorrect trajectories should show smaller distances than correct ones (negative Δ(Incorrect−Correct), non-overlapping CIs). Direction matters more than magnitude — you're checking incorrect < correct in movement.
 The gap should be larger for the final transition (last step→answer marker) than for the second-to-last→last transition — divergence should intensify as you get closer to the end.
Section 4.2 — Mid-reasoning correctness prediction
 Train the logistic regression correctness classifier per Appendix F (or a reasonably close approximation — sklearn LogisticRegression with CV over C is fine for a first pass).
 Late-step trajectory features: should land somewhere in the ballpark of AUC ~0.80–0.87 (their peak was 0.87 at layer 29, avg ~0.83 across layers).
 Early-step features (Step 1→2 only): should be much weaker, close to their ~0.61–0.63.
 Key pattern: late-step AUC should clearly and consistently beat early-step AUC. If they're similar, something's off in how you're constructing the late vs. early feature sets.
 Optional secondary check: does AUC roughly increase with layer depth, peaking somewhere in the later layers (they found peak around layer 29, near but not at the very last layer)?
Section 4.3 — Intervention gating

This one's more involved to replicate correctly, so treat it as lower priority if you're time-constrained. If you do it:

 Unconditional (always-on) token injection: accuracy should drop relative to baseline (no-intervention) — confirm direction of harm, not necessarily exact magnitude.
 Gated version (only intervening on examples flagged by your Section 4.2 predictor): the accuracy drop should shrink substantially compared to unconditional, moving toward neutral (~0 change) or a small gain.
 Check the flagged-example count is a similar order of magnitude to their ~12% of the test set — if your predictor is flagging 50% or 1%, your threshold calibration is off.
Section 5.1 — Trajectory-based steering
 Stratify your GSM8K test set by step count, and measure accuracy before/after the low-rank correction, separately per step-count bucket.
 Short chains (≤5 steps): expect near-zero net effect.
 Long chains (6–7 steps): expect a positive accuracy bump (their numbers: +7.6 and +7.7 points — yours may differ but should be positive and non-trivial, not negative or flat).
 Preservation rate (fraction of originally-correct examples that stay correct after intervention) should be high, ≥~95% — if you're seeing preservation much lower than that, your steering magnitude/threshold is too aggressive.
Section 5.2 — Length control
 Sweep |α| from small (e.g. 0.1) up to ~0.8+ for both SHORTEN and PROLONG.
 At small |α| (≤0.4), expect roughly monotonic, graded change in output length with minimal accuracy cost (~1% or less).
 At high |α| (beyond ~0.8), expect a qualitative breakdown — repetitive/looping generations that don't terminate normally. If you don't see any breakdown even at high α, your steering vector or intervention layers might be too weak to actually move the activation off-manifold (worth checking the vector norm/scale you're using).
General cross-checks to run early (before trusting any of the above)
 Confirm your Step-marker token-position extraction is correct by manually inspecting a handful of decoded examples — print the token right before each detected Step/answer marker and eyeball that it's the right spot.
 Confirm activation shapes are [n_layers=33, hidden=4096] and that you're pulling post-residual-add, pre-RMSNorm hidden states (this is just outputs.hidden_states[layer] from HF transformers, which already gives you this by default — just make sure you're not accidentally grabbing post-norm values from a custom hook).
 Confirm greedy decoding (do_sample=False) is actually being used — any sampling randomness will make your correctness labels and downstream comparisons noisier than the paper's deterministic setup.