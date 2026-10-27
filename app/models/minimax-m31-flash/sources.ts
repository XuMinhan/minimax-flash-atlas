export const VLLM_COMMIT = "ced6857afa0ea7b2e3f0846a62e1394e90f15607";
export const CODE_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/minimax_m3/nvidia/model.py`;
export const WEIGHTS_URL = "https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash";
export const RUNNER_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/v1/worker/gpu_model_runner.py`;
export const ACTIVATION_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/model_executor/layers/activation.py`;
export const LINEAR_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/model_executor/layers/linear.py`;
export const VLLM_INDEXER_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/minimax_m3/common/indexer.py`;
export const VLLM_SPARSE_ATTENTION_URL = `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/models/minimax_m3/common/sparse_attention.py`;
export const NORM_FORWARD_URL = `${CODE_URL}#L130-L142`;
export const DECODER_FORWARD_URL = `${CODE_URL}#L752-L778`;

export const HF_REVISION = "262ae6830934f3512211081b8a3f16425b93a239";
export const CHECKPOINT_URL = `${WEIGHTS_URL}/blob/${HF_REVISION}/config.json`;
export const DRAFT_CONFIG_URL = `${WEIGHTS_URL}/blob/${HF_REVISION}/dspark/config.json`;
