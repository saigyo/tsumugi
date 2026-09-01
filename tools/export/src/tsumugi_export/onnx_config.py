"""Custom ONNX export config: stock Llama-family causal-LM outputs plus per-layer attention weights (attentions.{i}) and the hidden-state chain (hidden_states.{i}, reduced to inputs_embeds after export).

API-shape caveat: verify the base-class import path and constructor signature
against the INSTALLED optimum version (optimum.exporters.onnx.model_configs
.LlamaOnnxConfig at the time of writing). Fix mismatches here; never change
the output names or axes — they are the artifact contract."""
from optimum.exporters.onnx.model_configs import LlamaOnnxConfig


class AttnLlamaOnnxConfig(LlamaOnnxConfig):
    @property
    def outputs(self):
        outs = dict(super().outputs)
        n = self._config.num_hidden_layers
        # torch.onnx.export assigns these names POSITIONALLY to the flattened
        # forward outputs (CausalLMOutputWithPast order: logits, past_key_values,
        # hidden_states, attentions). transformers returns n+1 hidden states with
        # index 0 = the embedding lookup; postprocess.expose_inputs_embeds keeps
        # that one as `inputs_embeds` and drops the rest.
        for i in range(n + 1):
            outs[f"hidden_states.{i}"] = {0: "batch_size", 1: "sequence_length"}
        for i in range(n):
            outs[f"attentions.{i}"] = {0: "batch_size", 2: "query_length", 3: "kv_length"}
        return outs
