from transformers import LlamaConfig

from tsumugi_export.onnx_config import AttnLlamaOnnxConfig


def make_config() -> LlamaConfig:
    return LlamaConfig(
        num_hidden_layers=30, num_attention_heads=9, num_key_value_heads=3,
        hidden_size=576, intermediate_size=1536, vocab_size=49152,
    )


def test_outputs_include_per_layer_attentions():
    cfg = AttnLlamaOnnxConfig(make_config(), task="text-generation", use_past=True)
    outs = cfg.outputs
    assert "logits" in outs
    for i in range(30):
        assert f"attentions.{i}" in outs
        assert outs[f"attentions.{i}"] == {0: "batch_size", 2: "query_length", 3: "kv_length"}


def test_cache_outputs_survive():
    cfg = AttnLlamaOnnxConfig(make_config(), task="text-generation", use_past=True)
    assert any(k.startswith("present") for k in cfg.outputs)


def test_no_cache_variant_has_no_present():
    cfg = AttnLlamaOnnxConfig(make_config(), task="text-generation", use_past=False)
    assert not any(k.startswith("present") for k in cfg.outputs)
    assert "attentions.0" in cfg.outputs


def test_use_past_in_inputs_true_yields_past_key_values_inputs():
    cfg = AttnLlamaOnnxConfig(
        make_config(), task="text-generation", use_past=True, use_past_in_inputs=True)
    ins = cfg.inputs
    assert "past_key_values.0.key" in ins
    assert "past_key_values.0.value" in ins


def test_without_use_past_in_inputs_no_past_key_values_inputs():
    # use_past controls the OUTPUT-side cache (present.*); it does not by
    # itself add past_key_values.* INPUTS — that requires the separate
    # use_past_in_inputs flag. export.py must pass both, or the exported
    # graph declares present.* outputs with no way to feed a cache back in.
    cfg = AttnLlamaOnnxConfig(make_config(), task="text-generation", use_past=True)
    ins = cfg.inputs
    assert not any(k.startswith("past_key_values") for k in ins)
