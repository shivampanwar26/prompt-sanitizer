import os
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import yaml


@dataclass
class SanitizerConfig:
    # ── Models ──────────────────────────────────────────────────
    causal_model: str = "distilgpt2"
    mask_model: str = "roberta-base"
    ner_model: str = "dslim/distilbert-NER"
    seq2seq_model: str = "google/flan-t5-small"

    # "prosan"   -> PII recognizers + NER + contextual (LM-scored) desensitization
    # "pii_only" -> recognizers + NER only; the causal and masked LMs are never loaded
    mode: str = "prosan"

    # ── Detection layer ─────────────────────────────────────────
    ner_backend: str = "transformer"   # "transformer", "nltk" or "none"
    ner_threshold: float = 0.80        # min entity confidence from the NER model
    ner_labels: Tuple[str, ...] = ("PER", "LOC", "ORG")
    propagate_entities: bool = True    # hide every mention of a detected entity

    # ── Surrogates ──────────────────────────────────────────────
    # "realistic"   -> consistent, type-preserving fake values (best readability)
    # "placeholder" -> numbered tags such as <PERSON_1> (exactly reversible)
    surrogate_style: str = "realistic"

    # ── Contextual selection (Section V-A) ──────────────────────
    lambda_scale: float = 0.3          # gamma_q = lambda * sigmoid(H_q): replacement budget
    min_privacy: float = 0.5           # absolute privacy-risk threshold on O_w in [0, 1]
    max_importance: float = 0.6        # protect words above this utility importance K_w
    privacy_saturation: float = 8.0    # bits at which surprisal risk reaches 1 - 1/e
    importance_method: str = "attention"  # "attention" (1 pass) or "gradient" (paper)
    pos_filter_enabled: bool = True

    # ── Replacement generation (Section V-B) ────────────────────
    eta: float = 1.0
    tau: float = 0.9
    top_k: int = 15
    sampling_mode: str = "sample"      # "sample" or "max"

    device: str = "auto"
    seed: Optional[int] = 42

    @classmethod
    def load_from_yaml(cls, yaml_path: str = "config/config.yaml") -> "SanitizerConfig":
        config = cls()
        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Sections are only for readability; every key maps to a flat field.
            flat = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    flat.update(value)
                else:
                    flat[key] = value
            if "privacy_ratio" in flat and "lambda_scale" not in flat:
                flat["lambda_scale"] = flat.pop("privacy_ratio")

            config = config.with_overrides(
                {k: v for k, v in flat.items() if k in cls.field_names()}
            )

        # Environment variable overrides
        if os.getenv("CAUSAL_MODEL"):
            config.causal_model = os.getenv("CAUSAL_MODEL")
        if os.getenv("MASK_MODEL"):
            config.mask_model = os.getenv("MASK_MODEL")
        if os.getenv("SANITIZER_DEVICE"):
            config.device = os.getenv("SANITIZER_DEVICE")

        return config

    @classmethod
    def field_names(cls):
        return {f.name for f in fields(cls)}

    def with_overrides(self, overrides) -> "SanitizerConfig":
        """Return a copy with ``overrides`` applied, coercing to each field's type."""
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
