import math
import re
from typing import List, Dict, Tuple
from .types import WordScore

try:
    import nltk
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


def _ensure_nltk_pos():
    if not _NLTK_AVAILABLE:
        return
    try:
        nltk.pos_tag(["test"])
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


# Code-structural tokens that must never be replaced because they carry
# functional meaning (library names, keywords, built-in identifiers).
# This is a lightweight heuristic: if the word looks like a Python/JS/SQL
# identifier, dotted module path, or well-known keyword, protect it.
_CODE_KEYWORD_RE = re.compile(
    r"^(?:"
    r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+"   # dotted module paths: psycopg2.connect, os.path
    r"|import|from|def|class|return|raise|try|except|if|else|elif|for|while"
    r"|print|self|None|True|False|int|str|float|list|dict|set|tuple"
    r"|SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|FROM|WHERE|JOIN"
    r"|function|const|let|var|require|async|await"
    r"|connection|cursor|execute|connect|commit|rollback|close|fetch"
    r"|database|db|table|schema|query|engine|session"
    r"|host|hostname|port|password|passwd|user|username|secret|token"
    r")$",
    re.I,
)


class WordSelector:
    """
    ProSan Adaptive Word Selection (Section V-A).

    1. Calculates prompt-level average self-information H_q = (1/k) * sum(I_wj).
    2. Dynamically calculates protection ratio gamma_q = lambda_scale * (1 / (1 + exp(-H_q))).
    3. Preserves high-importance words (K_w > max_importance) to maintain output utility & task intent.
    4. Protects code-structural tokens (library names, keywords, dotted identifiers).
    5. Filters out structural function words (prepositions, conjunctions, articles, pronouns) via POS tagging.
    6. Focuses desensitization on low-importance content words with high privacy self-information risk.
    """

    # Content word POS tag prefixes in Penn Treebank tagset (Nouns, Adjectives, Adverbs, Numbers)
    CONTENT_POS_PREFIXES = ("NN", "JJ", "RB", "CD")

    def __init__(self, lambda_scale=0.25, min_privacy=0.5, max_importance=0.75, pos_filter_enabled=True):
        self.lambda_scale = lambda_scale
        self.min_privacy = min_privacy
        self.max_importance = max_importance
        self.pos_filter_enabled = pos_filter_enabled
        self._pos_checked = False

    def _tag_pos(self, tokens: List[str]) -> List[str]:
        if not self.pos_filter_enabled or not _NLTK_AVAILABLE:
            return ["NN" for _ in tokens]

        if not self._pos_checked:
            _ensure_nltk_pos()
            self._pos_checked = True

        try:
            tagged = nltk.pos_tag(tokens)
            return [tag for _, tag in tagged]
        except Exception:
            return ["NN" for _ in tokens]

    @staticmethod
    def _is_code_structural(word: str) -> bool:
        """Return True if the word looks like a code keyword or structural identifier."""
        if _CODE_KEYWORD_RE.match(word):
            return True
        # Dotted paths not caught by the regex (e.g. very long chains)
        if "." in word and all(part.isidentifier() for part in word.split(".")):
            return True
        return False

    # Units that make an adjacent number factual/contextual data rather than PII.
    _UNIT_RE = re.compile(
        r"(?:°[FC]|[FC]$"             # temperature: °F, °C
        r"|mg|kg|lb|lbs|oz|g|ml|L"     # weight/volume
        r"|mm|cm|m|km|ft|in|inch"      # distance
        r"|mph|km/h|kph"               # speed
        r"|%|percent"                  # percentage
        r"|days?|hours?|hrs?|minutes?|mins?|seconds?|secs?|weeks?|months?|years?"  # time
        r"|bpm|mmHg|IU|mcg|mg/dL"      # medical units
        r")",
        re.I,
    )

    def _is_number_with_unit(self, item: WordScore, full_text: str) -> bool:
        """Return True if this word is a number adjacent to a measurement unit."""
        w = item.word
        # Only protect numeric-looking words
        if not re.match(r"^\d+\.?\d*$", w):
            return False
        if not full_text:
            return False
        # Check text immediately after the number for a unit
        after = full_text[item.end:item.end + 12]
        if self._UNIT_RE.match(after.lstrip()):
            return True
        # Also check for "unit NUMBER" patterns like "day 2"
        before = full_text[max(0, item.start - 12):item.start]
        if self._UNIT_RE.search(before.split()[-1]) if before.strip() else False:
            return True
        return False


    def select(
        self,
        words: List[Dict],
        importance: List[float],
        privacy: List[float],
        raw_privacy: List[float],
        full_text: str = "",
    ) -> Tuple[List[WordScore], float, float]:
        """
        Returns (selected_words, H_q, gamma_q)
        """
        if not words:
            return [], 0.0, 0.0

        # Calculate average self-information H_q over all words in prompt
        H_q = sum(raw_privacy) / len(raw_privacy) if raw_privacy else 0.0

        # Formula (7): gamma_q = lambda_scale * (1 / (1 + exp(-H_q)))
        gamma_q = self.lambda_scale * (1.0 / (1.0 + math.exp(-H_q)))

        tokens = [w["word"] for w in words]
        pos_tags = self._tag_pos(tokens)

        items = []
        for i, word in enumerate(words):
            items.append(
                WordScore(
                    word=word["word"],
                    start=word["start"],
                    end=word["end"],
                    importance=importance[i],
                    privacy=privacy[i],
                    raw_privacy=raw_privacy[i],
                    pos_tag=pos_tags[i],
                )
            )

        # Build eligible list: exclude PII placeholders, high-importance words, and code tokens
        eligible = []
        for item in items:
            w_str = item.word
            # Skip PII placeholder tokens
            if w_str.startswith("<") or w_str.endswith(">"):
                continue
            if w_str in {"EMAIL", "PHONE", "URL", "IP_ADDRESS", "API_KEY",
                         "CREDIT_CARD", "PASSWORD", "DB_USER", "DB_HOST",
                         "CARD_NUMBER", "DATE", "PERSON", "ADDRESS", "LOCATION"}:
                continue
            # Preserve high-importance words (K_w > max_importance)
            if item.importance > self.max_importance:
                continue
            # Protect code-structural tokens
            if self._is_code_structural(w_str):
                continue
            # Protect numbers that appear next to units (medical, measurement, time)
            if self._is_number_with_unit(item, full_text):
                continue
            eligible.append(item)

        # Filter by POS and privacy threshold
        # Filter by POS: only content words (nouns, adjectives, adverbs, numbers) are
        # eligible. If no content words pass, we select nothing — replacing verbs,
        # determiners, or prepositions would break grammar with no privacy gain.
        if self.pos_filter_enabled:
            candidates = [
                x for x in eligible
                if x.pos_tag.startswith(self.CONTENT_POS_PREFIXES)
                and x.privacy >= self.min_privacy
            ]
        else:
            candidates = [x for x in eligible if x.privacy >= self.min_privacy]

        # Sort candidates by ascending importance (lowest utility impact first)
        candidates.sort(key=lambda x: (x.importance, -x.privacy))

        # Protection ratio determines target count
        count = max(1, int(math.ceil(len(items) * gamma_q))) if candidates else 0
        selected = candidates[:count]

        return selected, H_q, gamma_q

