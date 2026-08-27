"""Custom ONNX export config: stock Llama-family causal-LM outputs plus
per-layer attention weights (attentions.{i}).

API-shape caveat: verify the base-class import path and constructor signature
against the INSTALLED optimum version (optimum.exporters.onnx.model_configs
.LlamaOnnxConfig at the time of writing). Fix mismatches here; never change
the output names or axes — they are the artifact contract."""
from optimum.exporters.onnx.model_configs import LlamaOnnxConfig


class AttnLlamaOnnxConfig(LlamaOnnxConfig):
    @property
    def outputs(self):
        outs = dict(super().outputs)
        for i in range(self._config.num_hidden_layers):
            outs[f"attentions.{i}"] = {0: "batch_size", 2: "query_length", 3: "kv_length"}
        return outs
