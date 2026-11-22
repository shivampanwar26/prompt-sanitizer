from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple


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

    @classmethod
    def field_names(cls):
        return {f.name for f in fields(cls)}

    def with_overrides(self, overrides: Dict[str, Any]) -> "SanitizerConfig":
        """Return a validated copy with ``overrides`` applied, coerced to each field's type."""
        unknown = set(overrides) - self.field_names()
        if unknown:
            raise ValueError(f"Unknown configuration override(s): {', '.join(sorted(unknown))}")

        values = {f.name: getattr(self, f.name) for f in fields(self)}
        for key, value in overrides.items():
            if value is None and key != "seed":
                continue
            current = values[key]
            if isinstance(current, bool):
                value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
            elif isinstance(current, float):
                value = float(value)
            elif isinstance(current, int) and value is not None:
                value = int(value)
            elif isinstance(current, tuple):
                value = tuple(value)
            values[key] = value
        config = type(self)(**values)
        config.validate()
        return config

    def validate(self) -> None:
        choices = {
            "mode": ("prosan", "pii_only"),
            "surrogate_style": ("realistic", "placeholder"),
            "ner_backend": ("transformer", "nltk", "none"),
            "importance_method": ("attention", "gradient"),
            "sampling_mode": ("sample", "max"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {', '.join(allowed)}; got {getattr(self, name)!r}")
