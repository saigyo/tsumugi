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
