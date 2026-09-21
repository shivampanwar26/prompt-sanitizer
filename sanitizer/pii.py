"""Deterministic detection of high-confidence personally identifiable information.

Language-model surprise is not a reliable signal for privacy: ordinary medical
words and uncommon phrasing can be surprising too.  These rules deliberately
cover values whose format is sufficient evidence that they are sensitive.
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PIISpan:
    start: int
    end: int
    kind: str

    @property
    def placeholder(self) -> str:
        return f"<{self.kind}>"


# ── Realistic fake replacements ────────────────────────────────
# Instead of leaving raw <TAG> placeholders, we substitute with
# plausible but clearly fake values that maintain sentence readability.

_FAKE_NAMES = [
    "Alex", "Jordan", "Sam", "Morgan", "Casey", "Riley",
    "Taylor", "Jamie", "Avery", "Quinn", "Drew", "Skyler",
]
_FAKE_CITIES = [
    "Springfield", "Riverside", "Fairview", "Greenville",
    "Madison", "Franklin", "Clinton", "Georgetown",
]
_FAKE_STREETS = [
    "123 Main Street", "456 Elm Avenue", "789 Park Drive",
    "100 Maple Road", "250 Cedar Lane", "88 Pine Court",
]
_FAKE_DATES = [
    "January 1st", "June 15th", "September 10th",
    "April 22nd", "November 3rd", "August 7th",
]


def get_fake_replacement(kind: str, seed_text: str = "") -> str:
    """Return a realistic fake value for a given PII kind.

    Uses a deterministic seed based on the original text so the same
    input always produces the same fake output (reproducibility).
    """
    rng = random.Random(hash(seed_text) & 0xFFFFFFFF)

    if kind == "PERSON":
        return rng.choice(_FAKE_NAMES)
    elif kind == "DATE":
        return rng.choice(_FAKE_DATES)
    elif kind == "LOCATION":
        return rng.choice(_FAKE_CITIES)
    elif kind == "ADDRESS":
        return rng.choice(_FAKE_STREETS)
    elif kind == "EMAIL":
        return "user@example.com"
    elif kind == "IP_ADDRESS":
        return "xxx.xxx.xxx.xxx"
    elif kind == "PASSWORD":
        return "********"
    elif kind == "DB_USER":
        return "db_user"
    elif kind == "DB_HOST":
        return "db.example.com"
    elif kind == "PHONE":
        return "XXX-XXX-XXXX"
    elif kind == "URL":
        return "https://example.com"
    elif kind == "API_KEY":
        return "sk-XXXXXXXXXXXX"
    elif kind == "CARD_NUMBER":
        return "XXXX-XXXX-XXXX-XXXX"
    else:
        return f"<{kind}>"


# ── Date patterns ──────────────────────────────────────────────
_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|October"
    r"|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)

_DATE_PATTERNS = [
    # "July 2nd", "March 15th", "Jan 3"
    re.compile(rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?\b", re.I),
    # "2nd July", "15 March"
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}\b", re.I),
    # "07/02/1995", "2024-01-15", "15.03.1990"
    re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b"),
    # "July 2nd, 1995"
    re.compile(rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s*\d{{4}}\b", re.I),
]

# ── Address patterns ───────────────────────────────────────────
_STREET_SUFFIXES = (
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|"
    r"Lane|Ln|Way|Place|Pl|Court|Ct|Circle|Cir|"
    r"Terrace|Ter|Trail|Trl|Parkway|Pkwy|Highway|Hwy)"
)

_ADDRESS_PATTERNS = [
    # "42 Oak Street", "123 Main Ave", "1600 Pennsylvania Avenue"
    re.compile(
        rf"\b\d{{1,5}}\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\s+{_STREET_SUFFIXES}\b",
        re.I,
    ),
]


# ── Standard regex patterns ────────────────────────────────────
_PATTERNS = (
    ("EMAIL", re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w-])", re.I)),
    ("URL", re.compile(r"\b(?:https?://|www\.)[^\s<>]+(?<![.,!?;:])", re.I)),
    ("IP_ADDRESS", re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b")),
    ("CARD_NUMBER", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]?){2,3}\d{3,4}(?!\w)")),
    ("API_KEY", re.compile(r"\b(?:sk|pk|api)[_-][A-Za-z0-9_-]{16,}\b", re.I)),

    # Credential values in code
    ("PASSWORD", re.compile(
        r"""(?:password|passwd|pwd|secret|token|auth_token|api_key|apikey|access_key|private_key)"""
        r"""[ \t]*[=:][ \t]*"""
        r"""(?:"""
        r"""(?P<q>["'])(?P<qval>.+?)(?P=q)"""
        r"""|"""
        r"""(?P<uval>\S+)"""
        r""")""",
        re.I,
    )),
    ("DB_USER", re.compile(
        r"""(?:user|username|db_user|dbuser)"""
        r"""[ \t]*[=:][ \t]*"""
        r"""(?:"""
        r"""(?P<q>["'])(?P<qval>.+?)(?P=q)"""
        r"""|"""
        r"""(?P<uval>[^\s,;)}\]#]+)"""
        r""")""",
        re.I,
    )),
    ("DB_HOST", re.compile(
        r"""(?:host|hostname|db_host|server)"""
        r"""[ \t]*[=:][ \t]*"""
        r"""(?:"""
        r"""(?P<q>["'])(?P<qval>.+?)(?P=q)"""
        r"""|"""
        r"""(?P<uval>[^\s,;)}\]#]+)"""
        r""")""",
        re.I,
    )),
)


def _credential_value_span(match: re.Match) -> tuple[int, int]:
    """Return the (start, end) of just the value part of a key=value credential match."""
    qval = match.group("qval")
    if qval is not None:
        return match.start("qval"), match.end("qval")
    uval = match.group("uval")
    if uval is not None:
        return match.start("uval"), match.end("uval")
    return match.start(), match.end()


_CREDENTIAL_KINDS = {"PASSWORD", "DB_USER", "DB_HOST"}


# ── NER-based entity detection ────────────────────────────────
try:
    import nltk as _nltk
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


def _ensure_nltk_ner():
    """Download NLTK resources needed for named entity recognition."""
    if not _NLTK_AVAILABLE:
        return
    for resource in [
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker", "maxent_ne_chunker_tab",
        "words", "punkt", "punkt_tab",
    ]:
        try:
            _nltk.data.find(f"taggers/{resource}" if "tagger" in resource
                           else f"chunkers/{resource}" if "chunker" in resource
                           else f"corpora/{resource}" if resource == "words"
                           else f"tokenizers/{resource}")
        except LookupError:
            try:
                _nltk.download(resource, quiet=True)
            except Exception:
                pass


def _find_named_entities(text: str) -> List[PIISpan]:
    """Use NLTK NER to detect PERSON and GPE (location) named entities."""
    if not _NLTK_AVAILABLE:
        return []

    _ensure_nltk_ner()

    try:
        tokens = _nltk.word_tokenize(text)
        tagged = _nltk.pos_tag(tokens)
        tree = _nltk.ne_chunk(tagged)
    except Exception:
        return []

    # Map NER labels to our PII kinds
    label_map = {"PERSON": "PERSON", "GPE": "LOCATION"}

    # Common English words NLTK NER frequently misclassifies as PERSON
    # when they appear capitalized (e.g. at sentence start).
    _FALSE_POSITIVE_NAMES = {
        "please", "what", "the", "this", "that", "how", "why", "when",
        "where", "which", "who", "can", "could", "would", "should",
        "will", "do", "does", "did", "have", "has", "had", "is", "are",
        "was", "were", "be", "been", "being", "it", "he", "she", "they",
        "we", "you", "i", "my", "your", "his", "her", "its", "our",
        "their", "not", "no", "yes", "if", "but", "and", "or", "so",
        "also", "just", "only", "even", "still", "now", "then", "here",
        "there", "very", "much", "more", "most", "some", "any", "all",
        "each", "every", "both", "few", "many", "several", "such",
        "import", "from", "class", "def", "return", "print", "raise",
        "try", "except", "finally", "for", "while", "break", "continue",
        "pass", "with", "as", "in", "of", "to", "at", "by", "on",
        "about", "above", "after", "before", "between", "through",
        "during", "until", "against", "into", "out", "up", "down",
        "off", "over", "under", "again", "further", "once",
    }

    spans = []
    for subtree in tree:
        if not hasattr(subtree, "label"):
            continue
        label = subtree.label()
        if label not in label_map:
            continue

        name = " ".join(word for word, tag in subtree.leaves())
        kind = label_map[label]

        # Filter false positives
        if name.lower() in _FALSE_POSITIVE_NAMES:
            continue
        # Single-character entities are noise
        if len(name) <= 1:
            continue

        # Find the entity's position in the original text
        idx = text.find(name)
        if idx >= 0:
            spans.append(PIISpan(idx, idx + len(name), kind))

    return spans


def find_pii(text: str) -> List[PIISpan]:
    """Return non-overlapping, high-confidence PII spans in source order."""
    candidates = []

    # Standard pattern-based PII
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            if kind in _CREDENTIAL_KINDS:
                start, end = _credential_value_span(match)
            else:
                start, end = match.start(), match.end()
            candidates.append(PIISpan(start, end, kind))

    # Date patterns
    for date_re in _DATE_PATTERNS:
        for match in date_re.finditer(text):
            candidates.append(PIISpan(match.start(), match.end(), "DATE"))

    # Address patterns (must come before NER so regex gets priority)
    for addr_re in _ADDRESS_PATTERNS:
        for match in addr_re.finditer(text):
            candidates.append(PIISpan(match.start(), match.end(), "ADDRESS"))

    # NER-based person + location detection
    candidates.extend(_find_named_entities(text))

    candidates.sort(key=lambda item: (item.start, -(item.end - item.start)))

    accepted: List[PIISpan] = []
    for item in candidates:
        if accepted and item.start < accepted[-1].end:
            continue
        accepted.append(item)
    return accepted
