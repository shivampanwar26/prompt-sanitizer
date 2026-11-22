from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SanitizerConfig:
    causal_model: str = "distilgpt2"
    mask_model: str = "roberta-base"
    ner_model: str = "dslim/distilbert-NER"

    mode: str = "prosan"

    ner_backend: str = "transformer"
    ner_threshold: float = 0.5
    ner_labels: Tuple[str, ...] = ("PER", "LOC", "ORG")
    propagate_entities: bool = True

    surrogate_style: str = "realistic"

    lambda_scale: float = 0.3
    min_privacy: float = 0.5
    max_importance: float = 0.6
    privacy_saturation: float = 8.0
    importance_method: str = "attention"
    pos_filter_enabled: bool = True

    top_k: int = 15
    eta: float = 1.0
    tau: float = 0.9
    sampling_mode: str = "sample"

    device: str = "auto"
    seed: Optional[int] = 42
