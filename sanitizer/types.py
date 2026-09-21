from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class WordScore:
    word: str
    start: int
    end: int
    importance: float
    privacy: float
    raw_privacy: float = 0.0
    pos_tag: str = ""
    replacement: Optional[str] = None


@dataclass
class SanitizationResult:
    text: str
    selected_words: List[Dict]
    H_q: float = 0.0
    gamma_q: float = 0.0
    perplexity: Optional[float] = None
    original_perplexity: Optional[float] = None
    phr: Optional[float] = None
    # surrogate -> original, for restoring an LLM response (see PromptSanitizer.restore)
    mapping: Dict[str, str] = field(default_factory=dict)
