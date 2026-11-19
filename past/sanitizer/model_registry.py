import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from typing import Dict, Tuple, Any

_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_causal_model_and_tokenizer(model_name: str, device: str = "auto") -> Tuple[Any, Any]:
    target_device = _resolve_device(device)

    cache_key = f"causal_{model_name}_{target_device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # FP16 on CUDA for ~2x speed + ~40% memory reduction
    dtype = torch.float16 if target_device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="eager",  # Required for output_attentions=True
    ).to(target_device)
    model.eval()

    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def get_mask_model_and_tokenizer(model_name: str, device: str = "auto") -> Tuple[Any, Any]:
    target_device = _resolve_device(device)

    cache_key = f"mask_{model_name}_{target_device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # FP16 on CUDA for ~2x speed + ~40% memory reduction
    dtype = torch.float16 if target_device.type == "cuda" else torch.float32

    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(target_device)
    model.eval()

    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer
