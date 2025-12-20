from transformers import PretrainedConfig


class GptOssConfig(PretrainedConfig):
    model_type = "gpt_oss"

    def __init__(
        self,
        num_hidden_layers: int = 36,
        num_experts: int = 128,
        experts_per_token: int = 4,
        vocab_size: int = 201088,
        hidden_size: int = 2880,
        intermediate_size: int = 2880,
        head_dim: int = 64,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        sliding_window: int = 128,
        initial_context_length: int = 4096,
        rope_theta: float = 150000.0,
        rope_scaling_factor: float = 32.0,
        rope_ntk_alpha: float = 1.0,
        rope_ntk_beta: float = 32.0,
        pad_token_id: int = 0,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            **kwargs
        )
        self.num_hidden_layers = num_hidden_layers
        self.num_experts = kwargs.get("num_local_experts", num_experts)
        self.experts_per_token = kwargs.get("num_experts_per_tok", experts_per_token)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.sliding_window = sliding_window
        self.initial_context_length = initial_context_length
        self.rope_theta = rope_theta
        rope_config = kwargs.get("rope_scaling", {})
        self.rope_scaling_factor = rope_config.get("factor", rope_scaling_factor)
        self.rope_ntk_alpha = rope_config.get("beta_slow", rope_ntk_alpha)
        self.rope_ntk_beta = rope_config.get("beta_fast", rope_ntk_beta)
        # Related Ticket: https://aws-neuron.atlassian.net/jira/software/projects/NS/boards/34/backlog?selectedIssue=NS-139
        # Fake HF configs to disable the code reading the sliding_window_size during max length validation.
        # v0.9.0 vLLM reads sliding_window settings from HF config of the model (presumably to setup K/V cache management for vLLM's own modeling code).
        # This causes a block allocation which fails for our current Neuron integration because we only have 1 block per TKG batch size.
        self.use_sliding_window = False
