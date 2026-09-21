import os

# This application uses PyTorch models exclusively.  Set this before any
# Transformers import so its optional TensorFlow/Keras integration is not
# imported (and cannot conflict with a system Keras installation).
os.environ.setdefault("USE_TF", "0")

from .sanitizer import PromptSanitizer
from .config import SanitizerConfig
from .evaluator import PromptEvaluator
from .types import SanitizationResult, WordScore
from .dataset import generate_dataset_from_file, generate_sanitized_dataset

__all__ = [
    "PromptSanitizer",
    "SanitizerConfig",
    "PromptEvaluator",
    "SanitizationResult",
    "WordScore",
    "generate_dataset_from_file",
    "generate_sanitized_dataset",
]