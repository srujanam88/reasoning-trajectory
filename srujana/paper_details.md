1. Datasets
Dataset	Source	Split(s) needed	Notes
GSM8K	openai/gsm8k (HF), config main	train (7,473), test (1,319)	Core dataset for everything
MATH-500	HuggingFaceH4/MATH-500 (HF)	all 500	Cross-task transfer (Sec 4.4)
MMLU	cais/mmlu (HF), all config	validation (1,531)	Cross-task transfer (Sec 4.4)

Pull all three via datasets.load_dataset — no manual downloading needed.

2. Models

Small test model (for pipeline debugging, not for real results):

meta-llama/Llama-3.2-1B-Instruct — same family/tokenizer conventions as the paper's models, fast enough to run the whole pipeline on CPU or a small GPU in seconds per example. Use this to validate every stage before touching the 8B models.

Actual paper models (Llama-3.1-8B family):

Base: meta-llama/Llama-3.1-8B
Instruct: meta-llama/Llama-3.1-8B-Instruct
R1-Distill: deepseek-ai/DeepSeek-R1-Distill-Llama-8B

All three share the same tokenizer/architecture (32 transformer layers, 4096 hidden dim), which is what makes cross-model probe transfer (Table 1/6) meaningful. All require a HF token + license acceptance for the Meta models.

3. Implementation steps

Step 0 — Environment
transformers, torch, datasets, scikit-learn, numpy, accelerate. Single GPU, bf16 inference.

Step 1 — Prompt template
Implement the exact fixed-form prompt from Appendix A (forces Step [n]: markers and a ####-style final-answer marker). Get this exactly right — the whole pipeline depends on being able to locate marker token positions reliably.

Step 2 — Generation (Pass 1)
Batch-generate CoT solutions with model.generate(), greedy decoding (do_sample=False), KV cache on. Parse out the token index of each Step k: occurrence and the final-answer marker using the tokenizer's offset mapping — don't regex on decoded text and re-tokenize, get positions from the original generation to avoid alignment bugs.

Step 3 — Hidden state extraction (Pass 2)
Re-run the full prompt+completion through the model once with output_hidden_states=True, use_cache=False, single forward pass (not autoregressive). Pull the residual-stream activation (post-add, pre-RMSNorm — this is just hidden_states[layer] output from HF) at position t(Step k) - 1 for each step and t(term) - 1. Store as [n_examples, n_layers=33, n_positions, 4096], fp16, to disk (e.g. memory-mapped numpy or HDF5) — don't keep it all in RAM.

Step 4 — Correctness labeling
Extract the final numeric answer from generated text, compare to ground truth (GSM8K/MATH answers are parseable numbers; MMLU is multiple choice). Store as a binary label per example, aligned with the activation cache from Step 3.

Step 5 — Step-identity linear probes (Section 3)
For each layer × each step label, train logistic regression (one-vs-rest, class_weight='balanced') on that layer's activations, 80/20 stratified split. Reproduce Fig. 1b curves and the cross-model transfer table (train probe on model A's activations, eval on model B's, same layer).

Step 6 — Trajectory distance analysis (Section 4.1)
Compute Euclidean and cosine distances between consecutive step activations at the final layer, split by correctness label, bootstrap 95% CIs. This just needs Step 3's cached activations — no new model calls.

Step 7 — Correctness predictor (Section 4.2)
Build the feature sets (early-step, late-step+PCA-128, final-state) and train the single-layer logistic regression classifier described in Appendix F (Adam, weight_decay=1/C, 5-fold CV over C). Report test ROC-AUC per layer, compare against step-count-only and logit-lens baselines.

Step 8 — Steering vectors + gated intervention (Section 4.3)
Compute s = mean(h_term - h_step_k) per layer on the training split. At inference, hook the model to add α·s at the target layers (LAST=27–31, MID=13–17) at the token position right before the answer marker. Wrap generation so the correctness predictor from Step 7 gates whether the intervention fires. This needs a forward_pre_hook or similar on the relevant decoder layers.

Step 9 — Trajectory-based steering (Section 5.1)
Fit PCA-128 on correct-example final-layer activations, compute step-wise mean μⱼ and dispersion σⱼ. At inference, project the current activation, compute local/cumulative deviation, and apply a rank-32 correction toward μⱼ when thresholds are exceeded. This is the most involved intervention — build and validate it last, after Steps 5–8 work.

Step 10 — Reasoning-length control
Reuse Step 8's steering machinery with sign flipped (SHORTEN vs PROLONG) and sweep |α|; measure length change and accuracy.

Recommended order: get Steps 0–4 fully working on the 1B model + a 20–50 example GSM8K subsample first. Only once that pipeline is verified end-to-end (correct marker alignment, correct activation shapes, sane probe accuracies) should you swap in the 8B models and full datasets.