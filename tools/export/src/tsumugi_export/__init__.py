STOCK_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
ATTN_REPO_ID = "saigyo-hoshi/smollm2-135m-attn-onnx"
# Candidate rows of E gathered inside the graph at the last position. The app
# caps top-k at 10; 16 leaves headroom for ties.
TOPK = 16
