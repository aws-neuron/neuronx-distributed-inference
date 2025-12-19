import types
import transformers

from packaging.version import parse as parse_version

from transformers import Llama4TextModel


def patch_llama4_text_moe_forward(model: Llama4TextModel):
    """
    Patches Llama4TextMoe modules to fix an accuracy issue that affects transformers v4.54-4.56.

    Context: https://github.com/huggingface/transformers/pull/40609
    """

    # TODO: Add upper version constraint once the issue is fixed in transformers.
    if parse_version(transformers.__version__) < parse_version("4.54.0"):
        return

    # Fixed forward function for Llama4TextMoe.
    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_scores, router_logits = self.router(hidden_states)
        routed_in = hidden_states.repeat(router_scores.shape[1], 1)
        routed_in = routed_in * router_scores.transpose(0, 1).reshape(-1, 1)
        routed_out = self.experts(routed_in)
        out = self.shared_expert(hidden_states)
        out.add_(routed_out.reshape(router_scores.shape[1], -1, routed_out.shape[-1]).sum(dim=0))
        return out, router_logits

    for layer in model.layers:
        if layer.is_moe_layer:
            layer.feed_forward.forward = types.MethodType(forward, layer.feed_forward)
