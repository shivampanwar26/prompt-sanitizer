import os
from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class SanitizerConfig:
    causal_model: str = "distilgpt2"
    mask_model: str = "roberta-base"
    lambda_scale: float = 0.15   # Fraction of words to desensitize (conservative)
    min_privacy: float = 0.75    # Only target words with high privacy risk
    max_importance: float = 0.45  # Preserve words above this importance for utility
    eta: float = 1.0
    tau: float = 0.9
    top_k: int = 15
    sampling_mode: str = "sample"  # "sample" or "max"
    device: str = "auto"
    pos_filter_enabled: bool = True
    seed: Optional[int] = 42
    mode: str = "prosan"
    seq2seq_model: str = "google/flan-t5-small"

    @classmethod
    def load_from_yaml(cls, yaml_path: str = "config/config.yaml") -> "SanitizerConfig":
        config = cls()
        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            if "causal_model" in data:
                config.causal_model = data["causal_model"]
            if "mask_model" in data:
                config.mask_model = data["mask_model"]

            selector_cfg = data.get("selector", {})
            if "lambda_scale" in selector_cfg:
                config.lambda_scale = float(selector_cfg["lambda_scale"])
            elif "privacy_ratio" in selector_cfg:
                config.lambda_scale = float(selector_cfg["privacy_ratio"])
            if "min_privacy" in selector_cfg:
                config.min_privacy = float(selector_cfg["min_privacy"])
            if "max_importance" in selector_cfg:
                config.max_importance = float(selector_cfg["max_importance"])
            if "pos_filter_enabled" in selector_cfg:
                config.pos_filter_enabled = bool(selector_cfg["pos_filter_enabled"])

            replacement_cfg = data.get("replacement", {})
            if "top_k" in replacement_cfg:
                config.top_k = int(replacement_cfg["top_k"])
            if "eta" in replacement_cfg:
                config.eta = float(replacement_cfg["eta"])
            if "tau" in replacement_cfg:
                config.tau = float(replacement_cfg["tau"])
            if "sampling_mode" in replacement_cfg:
                config.sampling_mode = str(replacement_cfg["sampling_mode"])

            if "device" in data:
                config.device = data["device"]
            if "seed" in data:
                config.seed = data["seed"]
            if "mode" in data:
                config.mode = str(data["mode"])
            if "seq2seq_model" in data:
                config.seq2seq_model = str(data["seq2seq_model"])

        # Environment variable overrides
        if os.getenv("CAUSAL_MODEL"):
            config.causal_model = os.getenv("CAUSAL_MODEL")
        if os.getenv("MASK_MODEL"):
            config.mask_model = os.getenv("MASK_MODEL")

        return config
