import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from typing import Dict, Tuple, Any

_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


def resolve_device(device: str) -> torch.device:
    if device == "auto" or (device.startswith("cuda") and not torch.cuda.is_available()):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_causal_model_and_tokenizer(model_name: str, device: str = "auto") -> Tuple[Any, Any]:
    target_device = resolve_device(device)

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
    target_device = resolve_device(device)

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


def get_vocab_masks(model_name: str, tokenizer, device: torch.device) -> Dict[str, torch.Tensor]:
    """Boolean masks over the MLM vocabulary for whole-word replacement candidates.

    Keys are "<ws|nows>_<upper|lower>": whether the token starts a new word
    after whitespace (RoBERTa "Ġ", SentencePiece "▁", or any non-"##" WordPiece
    token) and whether it is capitalised.  Only alphabetic tokens of length >= 2
    qualify, so punctuation, digits and sub-word fragments are never proposed.
    """
    cache_key = f"vocab_{model_name}_{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    size = len(tokenizer)
    masks = {k: torch.zeros(size, dtype=torch.bool) for k in ("ws_upper", "ws_lower", "nows_upper", "nows_lower")}
    wordpiece = any(t.startswith("##") for t in list(tokenizer.get_vocab())[:5000])
    special = set(tokenizer.all_special_ids)

    for token, idx in tokenizer.get_vocab().items():
        if idx >= size or idx in special:
            continue
        if token.startswith(("Ġ", "▁")):
            ws, text = True, token[1:]
        elif wordpiece:
            if token.startswith("##"):
                continue
            ws, text = True, token
        else:
            ws, text = False, token
        if len(text) < 2 or not text.isalpha() or not text.isascii():
            continue
        case = "upper" if text[0].isupper() else "lower"
        masks[f"{'ws' if ws else 'nows'}_{case}"][idx] = True

    masks = {k: v.to(device) for k, v in masks.items()}
    _MODEL_CACHE[cache_key] = masks
    return masks
