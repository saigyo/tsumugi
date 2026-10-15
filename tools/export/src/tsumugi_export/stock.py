"""Stock SmolLM2 weights and tokenizer, for the geometry asset and validation."""
import numpy as np

from tsumugi_export import STOCK_MODEL_ID


def stock_embeddings() -> np.ndarray:
    """The input embedding matrix, fp32 [vocab, dims]. Read straight from the
    safetensors file (via torch: the checkpoint is bf16, which numpy lacks)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    path = hf_hub_download(STOCK_MODEL_ID, "model.safetensors")
    return load_file(path)["model.embed_tokens.weight"].float().numpy()


def stock_token_texts(vocab: int) -> list[str]:
    """decode([id]) for every id — the same convention the app uses for TokenInfo.text."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STOCK_MODEL_ID)
    return [tok.decode([i]) for i in range(vocab)]


def _stock_state() -> dict:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    return load_file(hf_hub_download(STOCK_MODEL_ID, "model.safetensors"))


def stock_mlp_down() -> list[np.ndarray]:
    """Every layer's down_proj weight, fp32 [dims, neurons], in layer order."""
    state = _stock_state()
    out, l = [], 0
    while (key := f"model.layers.{l}.mlp.down_proj.weight") in state:
        out.append(state[key].float().numpy())
        l += 1
    return out


def stock_final_norm() -> np.ndarray:
    """The final RMSNorm's weight g, fp32 [dims]."""
    return _stock_state()["model.norm.weight"].float().numpy()
