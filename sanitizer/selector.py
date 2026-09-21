import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from .privacy import is_sentence_initial
from .types import WordScore


# Code-structural tokens that must never be replaced because they carry
# functional meaning (library names, keywords, built-in identifiers).
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

# Capitalised words that are public vocabulary, not identities.
_PUBLIC_PROPER = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "english", "hindi", "python",
    "java", "javascript", "typescript", "linux", "windows", "android", "ios", "sql",
    "postgres", "postgresql", "mysql", "docker", "kubernetes", "react", "google",
    "covid", "christmas", "diwali", "god", "internet", "excel", "word", "dr", "mr",
    "mrs", "ms", "prof", "ai", "gpt", "llm", "api",
    # Languages, nationalities and other public adjectives that are capitalised
    # in English but do not identify anyone.
    "french", "spanish", "german", "italian", "chinese", "japanese", "korean",
    "russian", "arabic", "portuguese", "dutch", "swedish", "greek", "latin",
    "european", "asian", "african", "australian", "canadian", "mexican",
    "christian", "muslim", "hindu", "buddhist", "jewish", "roman", "victorian",
}

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by",
    "for", "with", "from", "as", "is", "are", "was", "were", "be", "been", "it",
    "this", "that", "these", "those", "i", "me", "my", "you", "your", "he", "she",
    "his", "her", "we", "our", "they", "their", "what", "which", "who", "how",
    "why", "when", "where", "not", "no", "yes", "do", "does", "did", "have", "has",
    "had", "can", "could", "should", "would", "will", "please", "also", "very",
}

# Units that make an adjacent number factual/contextual data rather than PII.
_UNIT_RE = re.compile(
    r"(?:°[FC]?|[FC]\b"
    r"|mg|kg|lbs?|oz|g|ml|L\b|mcg|IU|mg/dL|mmHg|bpm"
    r"|mm|cm|m\b|km|ft|in\b|inch(?:es)?|mph|km/h|kph"
    r"|%|percent|x\b|times"
    r"|days?|hours?|hrs?|minutes?|mins?|seconds?|secs?|weeks?|months?|years?|yrs?)",
    re.I,
)
_UNIT_BEFORE_RE = re.compile(r"(?:day|week|month|year|step|stage|grade|type|level|version|v|chapter|page|room|floor)\s*$", re.I)


@dataclass
class SelectionUnit:
    """One entity to desensitize, covering every mention of it in the prompt."""
    text: str
    occurrences: List[Tuple[int, int]]
    importance: float
    privacy: float
    raw_privacy: float
    pos_tag: str
    members: List[WordScore] = field(default_factory=list)
    replacement: str = ""


class WordSelector:
    """
    ProSan Adaptive Word Selection (Section V-A) — Enhanced.

    1. H_q = mean self-information; budget gamma_q = lambda * sigmoid(H_q) (Eq. 7).
    2. Eligibility: content words (POS) that are not code tokens, measurement
       numbers, stop words or public proper nouns.  Words with utility
       importance above ``max_importance`` are protected unless they look like
       identities (proper-noun prior), because names are almost never what a
       task depends on.
    3. Absolute threshold O_w >= min_privacy (calibrated scores, see privacy.py);
       the gamma_q budget is an upper bound, not a quota, so benign prompts are
       left untouched.
    4. Selection is per *entity*: every mention of a selected word is selected,
       and adjacent selected capitalised words are merged ("Estrella Garcia")
       so they are replaced as one unit.
    5. Priority is privacy minus a utility penalty: O_w - 0.5 * K_w.
    """

    CONTENT_POS_PREFIXES = ("NN", "JJ", "RB", "CD", "FW")

    def __init__(self, lambda_scale=0.3, min_privacy=0.5, max_importance=0.6, pos_filter_enabled=True):
        self.lambda_scale = lambda_scale
        self.min_privacy = min_privacy
        self.max_importance = max_importance
        self.pos_filter_enabled = pos_filter_enabled

    @staticmethod
    def _is_code_structural(word: str) -> bool:
        if _CODE_KEYWORD_RE.match(word):
            return True
        if "." in word and all(part.isidentifier() for part in word.split(".")):
            return True
        return bool(re.search(r"_|[a-z][A-Z]", word))   # snake_case / camelCase identifiers

    @staticmethod
    def _is_number_with_unit(item: WordScore, full_text: str) -> bool:
        if not re.fullmatch(r"\d+(?:\.\d+)?", item.word) or not full_text:
            return False
        after = full_text[item.end:item.end + 12].lstrip()
        if _UNIT_RE.match(after):
            return True
        return bool(_UNIT_BEFORE_RE.search(full_text[max(0, item.start - 12):item.start]))

    def _eligible(self, item: WordScore, full_text: str) -> bool:
        w = item.word
        lower = w.lower()
        if lower in _STOPWORDS or lower in _PUBLIC_PROPER or len(w) < 2 and not w.isdigit():
            return False
        if self._is_code_structural(w) or self._is_number_with_unit(item, full_text):
            return False
        # Sentence-initial capitalisation is English orthography, not an identity
        # signal — "Translate this" must not get the same pass as "Estrella said".
        capitalized = w[:1].isupper() and not is_sentence_initial(full_text, item.start)
        looks_like_identity = capitalized or any(ch.isdigit() for ch in w)
        if self.pos_filter_enabled and not looks_like_identity and not item.pos_tag.startswith(self.CONTENT_POS_PREFIXES):
            return False
        if item.importance > self.max_importance and not looks_like_identity:
            return False
        return item.privacy >= self.min_privacy

    def select(
        self,
        words: List[Dict],
        importance: List[float],
        privacy: List[float],
        raw_privacy: List[float],
        pos_tags: List[str],
        full_text: str = "",
    ) -> Tuple[List[SelectionUnit], float, float]:
        """Returns (selection units, H_q, gamma_q)."""
        if not words:
            return [], 0.0, 0.0

        H_q = sum(raw_privacy) / len(raw_privacy)
        gamma_q = self.lambda_scale * (1.0 / (1.0 + math.exp(-H_q)))

        items = [
            WordScore(word=w["word"], start=w["start"], end=w["end"], importance=importance[i],
                      privacy=privacy[i], raw_privacy=raw_privacy[i], pos_tag=pos_tags[i])
            for i, w in enumerate(words)
        ]

        # Rank distinct surface forms; the budget counts entities, not mentions.
        by_form: Dict[str, List[WordScore]] = {}
        for item in items:
            by_form.setdefault(item.word.lower(), []).append(item)

        ranked = []
        for form, mentions in by_form.items():
            if any(self._eligible(m, full_text) for m in mentions):
                best = max(mentions, key=lambda m: m.privacy)
                min_imp = min(m.importance for m in mentions)
                ranked.append((best.privacy - 0.5 * min_imp, form))
        ranked.sort(reverse=True)

        budget = max(1, int(math.ceil(len(items) * gamma_q))) if ranked else 0
        chosen = {form for _, form in ranked[:budget]}
        selected = [item for item in items if item.word.lower() in chosen]

        return self._merge_units(selected, full_text), H_q, gamma_q

    @staticmethod
    def _merge_units(selected: List[WordScore], full_text: str) -> List[SelectionUnit]:
        # Merge runs like "Estrella Garcia" (capitalised, separated by one space).
        runs: List[List[WordScore]] = []
        for item in sorted(selected, key=lambda x: x.start):
            prev = runs[-1][-1] if runs else None
            if (prev is not None and item.word[:1].isupper() and prev.word[:1].isupper()
                    and full_text[prev.end:item.start] == " "):
                runs[-1].append(item)
            else:
                runs.append([item])

        units: Dict[str, SelectionUnit] = {}
        for run in runs:
            start, end = run[0].start, run[-1].end
            text = full_text[start:end] if full_text else " ".join(m.word for m in run)
            key = text.lower()
            if key not in units:
                units[key] = SelectionUnit(
                    text=text, occurrences=[], importance=max(m.importance for m in run),
                    privacy=max(m.privacy for m in run), raw_privacy=sum(m.raw_privacy for m in run),
                    pos_tag=run[0].pos_tag,
                )
            units[key].occurrences.append((start, end))
            units[key].members.extend(run)
        return list(units.values())
