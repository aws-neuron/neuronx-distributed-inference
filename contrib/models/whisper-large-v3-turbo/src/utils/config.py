from whisper.model import ModelDimensions


LargeV3Turbo = ModelDimensions(
    n_mels=128,
    n_audio_ctx=1500,
    n_audio_state=1280,
    n_audio_head=20,
    n_audio_layer=32,
    n_vocab=51866,
    n_text_ctx=448,
    n_text_state=1280,
    n_text_head=20,
    n_text_layer=4,
)


def get_dims_from_config(config) -> ModelDimensions:
    return ModelDimensions(
        n_mels=config.num_mel_bins,
        n_audio_ctx=config.max_source_positions,
        n_audio_state=config.d_model,
        n_audio_head=config.encoder_attention_heads,
        n_audio_layer=config.encoder_layers,
        n_vocab=config.vocab_size,
        n_text_ctx=config.max_target_positions,
        n_text_state=config.d_model,
        n_text_head=config.decoder_attention_heads,
        n_text_layer=config.decoder_layers,
    )
