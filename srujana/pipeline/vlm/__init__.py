"""VLM front-end for the trajectory pipeline (Qwen3-VL, multimodal datasets).

Adapts only the extraction front-end (model loading, prompting, image-aware two-pass
extraction, per-dataset labeling). It writes the SAME activations.dat / index.parquet schema
as the text pipeline, so the shared analysis layer (features, probes, distances, predictor,
viz) is reused unchanged. See srujana/VLM_PLAN.md.
"""
