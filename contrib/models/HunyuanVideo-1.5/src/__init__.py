"""
HunyuanVideo-1.5 on AWS Neuron (NxDI Contrib).

Text-to-video generation using the 8.33B DiT transformer from Tencent.
Runs on trn2.3xlarge with TP=4 and NKI flash attention.

Components:
  - dit_tp_wrapper: TP-sharded DiT backbone with NKI flash attention
  - dit_wrapper: CPU preprocessor for tracing
  - e2e_pipeline: Full text-to-video pipeline
  - compile_vae_neuron: VAE compilation with monkey-patches
  - tiled_vae_decode: Tiled VAE decode runtime
  - trace_byt5: byT5 text encoder tracing
  - cache_neg_embeddings: Negative embedding pre-cache for CFG
  - recompile_dit_masked: DiT compilation script
"""
