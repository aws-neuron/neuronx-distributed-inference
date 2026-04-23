"""YuE Music Generation on AWS Neuron (NxDI).

Full-song music generation from lyrics using YuE (M-A-P/HKUST).
Two-stage pipeline: S1 (7B LLaMA, TP=2) generates coarse codec tokens,
S2 (1B LLaMA, TP=1) refines via teacher-forcing, xcodec_mini decodes to audio.

Each stage runs in a separate subprocess (NxDI models with different TP degrees
cannot coexist in the same process).
"""
