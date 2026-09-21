"""High-precision detection of personally identifiable information and secrets.

Language-model surprise is not a reliable signal on its own: ordinary medical
words and uncommon phrasing are surprising too.  This module finds values
whose *form* or *context* is sufficient evidence that they are sensitive:

1. Validated pattern recognizers – a regex proposes, a checksum or parser
   disposes (Luhn for cards, mod-97 for IBAN, Verhoeff for Aadhaar,
   ``ipaddress`` for IPs, Shannon entropy for generic secrets).
2. Context recognizers – "my name is X", "Dr. X", "password is X",
   "patient ID: X" – which catch values no format rule can.
3. Named-entity recognition – a transformer NER model (PER / LOC / ORG),
   with NLTK's chunker as an offline fallback, filtered against code
   identifiers so ``ImportError`` or ``psycopg2`` are never "people".
4. Entity propagation – once "Sarah Connor" is found, every later "Sarah",
   "Connor" or "Sarah Connor" is hidden too (the paper's repeated-entity
   problem, solved at the span level rather than only by score pooling).
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PIISpan:
    start: int
    end: int
    kind: str
    score: float = 1.0
    source: str = "pattern"

    @property
    def placeholder(self) -> str:
        return f"<{self.kind}>"


# Overlap resolution: when spans collide, the higher priority wins, then the longer.
_PRIORITY = {
    "PRIVATE_KEY": 100, "JWT": 98, "API_KEY": 97, "EMAIL": 95, "URL": 90,
    "CARD_NUMBER": 88, "IBAN": 88, "SSN": 87, "AADHAAR": 87, "PAN": 86,
    "IP_ADDRESS": 85, "MAC_ADDRESS": 85, "PASSWORD": 80, "SECRET": 79,
    "USERNAME": 76, "HOSTNAME": 75, "PHONE": 70, "DATE": 65, "ADDRESS": 60,
    "ID_NUMBER": 58, "POSTAL_CODE": 57, "PERSON": 50, "LOCATION": 45,
    "ORGANIZATION": 40,
}


# ── Validators ──────────────────────────────────────────────────

def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def luhn_valid(value: str) -> bool:
    digits = _digits(value)
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def iban_valid(value: str) -> bool:
    compact = value.replace(" ", "").upper()
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(value: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(_digits(value))):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def ip_valid(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def shannon_entropy(value: str) -> float:
    counts = Counter(value)
    n = len(value)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _looks_like_secret(value: str) -> bool:
    """Random-looking token: long, mixed letters+digits, high character entropy."""
    if len(value) < 20 or not re.search(r"\d", value) or not re.search(r"[A-Za-z]", value):
        return False
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):  # git/sha hashes are rarely secrets
        return False
    return shannon_entropy(value) >= 3.5


_PHONE_CONTEXT = re.compile(
    r"(?:phone|mobile|cell|call|tel|telephone|contact|whatsapp|text|sms|fax|number|no\.)\W{0,3}(?:\w+\W{1,3}){0,3}$",
    re.I,
)


def _phone_ok(text: str, start: int, end: int) -> bool:
    value = text[start:end]
    digits = _digits(value)
    if not 7 <= len(digits) <= 15 or len(set(digits)) == 1:
        return False
    after = text[end:end + 6].lower()
    if re.match(r"\s*(?:mg|kg|ml|km|cm|mm|°|%|years?|yrs?|days?)\b", after):
        return False
    if value.startswith("+") or len(digits) >= 10:
        return True
    if re.fullmatch(r"\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}|\d{3}[.-]\d{4}", value.strip()):
        return True
    return bool(_PHONE_CONTEXT.search(text[max(0, start - 40):start]))


# ── Pattern recognizers ─────────────────────────────────────────

_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|October"
    r"|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)
_STREET_SUFFIXES = (
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|Way|Place|Pl"
    r"|Court|Ct|Circle|Cir|Terrace|Ter|Trail|Trl|Parkway|Pkwy|Highway|Hwy|Marg|Nagar)"
)
_KV_QUOTE = r"['\"]?"  # allows a quoted dict/JSON key: {'password': 'x'}
_KV_SEP = _KV_QUOTE + r"[ \t]*[=:][ \t]*"
_KV_SEP_STRICT = _KV_QUOTE + r"(?:[ \t]*=[ \t]*|:)"  # "User: Hi" in a chat log is not a credential
_KV_VALUE = r"""(?:(?P<q>["'])(?P<qval>[^"'\n]+?)(?P=q)|(?P<uval>[^\s,;"'`]+))"""

# (kind, regex, validator(text, match) -> bool, value_group)
_Recognizer = Tuple[str, "re.Pattern[str]", Optional[Callable[[str, "re.Match[str]"], bool]], Optional[str]]

_RECOGNIZERS: List[_Recognizer] = [
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"), None, None),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), None, None),
    ("API_KEY", re.compile(
        r"\b(?:(?:sk|pk|rk)[-_](?:live[-_]|test[-_]|proj[-_])?[A-Za-z0-9_-]{16,}"
        r"|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{10,}"
        r"|AIza[0-9A-Za-z_-]{35}|hf_[A-Za-z0-9]{30,})"), None, None),
    ("EMAIL", re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w-])", re.I), None, None),
    ("URL", re.compile(r"\b(?:https?://|www\.)[^\s<>\"')\]]+(?<![.,!?;:])", re.I), None, None),
    ("IP_ADDRESS", re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.]*\w)"),
     lambda t, m: ip_valid(m.group(0)), None),
    ("IP_ADDRESS", re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{1,4}){0,6}(?![\w:])"),
     lambda t, m: m.group(0).count(":") >= 2 and ip_valid(m.group(0)) and bool(re.search(r"[A-Fa-f]|::", m.group(0))), None),
    ("MAC_ADDRESS", re.compile(r"\b[0-9A-Fa-f]{2}(?:[:-][0-9A-Fa-f]{2}){5}\b"), None, None),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b"),
     lambda t, m: iban_valid(m.group(0)), None),
    ("CARD_NUMBER", re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])"),
     lambda t, m: luhn_valid(m.group(0)), None),
    ("SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), None, None),
    ("AADHAAR", re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)"),
     lambda t, m: verhoeff_valid(m.group(0)), None),
    ("PAN", re.compile(r"\b[A-Z]{3}[PCHABGJLFT][A-Z]\d{4}[A-Z]\b"), None, None),
    # Month names are case-sensitive so "you may 2" is not a date.
    ("DATE", re.compile(rf"\b{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s*\d{{4}})?\b"), None, None),
    ("DATE", re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTHS}\b(?:,?\s*\d{{4}})?"), None, None),
    # Dotted dates need a 4-digit year so version numbers ("3.10.12") are not dates.
    ("DATE", re.compile(r"\b\d{4}([-/.])\d{1,2}\1\d{1,2}\b|\b\d{1,2}([/-])\d{1,2}\2(?:\d{4}|\d{2})\b|\b\d{1,2}\.\d{1,2}\.\d{4}\b"), None, None),
    ("PHONE", re.compile(
        r"(?<![\w+.-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?)?(?:\d{7,15}|\d{2,5}(?:[\s.-]\d{2,5}){1,4})(?!\w)(?![.-]\d)"),
     lambda t, m: _phone_ok(t, m.start(), m.end()), None),
    ("ADDRESS", re.compile(rf"\b\d{{1,5}}(?:[A-Z])?,?\s+(?:[A-Z][a-z]+\s+){{1,3}}{_STREET_SUFFIXES}\b\.?"), None, None),
    ("ADDRESS", re.compile(r"\b(?:Flat|House|Apt|Apartment|Suite|Unit|Plot)\.?\s*(?:No\.?\s*)?#?\s*\d+[A-Z]?\b", re.I), None, None),
    ("POSTAL_CODE", re.compile(r"(?i:\b(?:zip|zip\s*code|postal\s*code|post\s*code|pin\s*code|pincode))\W{1,3}(?P<uval>[A-Z0-9]{3,4}\s?[A-Z0-9]{3}|\d{5}(?:-\d{4})?|\d{6})\b"), None, "uval"),
    ("ID_NUMBER", re.compile(
        r"(?i:\b(?:patient|mrn|medical\s+record|account|acct|policy|member|employee|customer|passport|"
        r"licen[cs]e|driver'?s?\s+licen[cs]e|ssn|aadhaar|roll|order|invoice|case|ticket|claim|student|"
        r"voter|pan|social\s+security)(?:\s+(?:id|number|no\.?|#))?)\s*(?:is\s+|:|#|-)?\s*"
        r"(?P<uval>(?=[A-Z0-9-]*\d)[A-Z0-9][A-Z0-9-]{3,})\b"), None, "uval"),
    # Credentials in code / config: only the value is sensitive.
    ("PASSWORD", re.compile(
        r"(?i:\b(?:password|passwd|pwd|pass|passcode|db_pass(?:word)?))" + _KV_SEP + _KV_VALUE), None, "value"),
    ("SECRET", re.compile(
        r"(?i:\b(?:secret|secret_key|client_secret|token|auth_token|access_token|api_key|apikey|access_key|private_key))"
        + _KV_SEP + _KV_VALUE), None, "value"),
    ("USERNAME", re.compile(r"(?i:\b(?:user|username|login|db_user|dbuser|uid))" + _KV_SEP_STRICT + _KV_VALUE), None, "value"),
    ("HOSTNAME", re.compile(r"(?i:\b(?:host|hostname|server|db_host|dbhost))" + _KV_SEP + _KV_VALUE), None, "value"),
    # Credentials in prose: "my password is hunter2", "the OTP is 482913".
    ("PASSWORD", re.compile(r"(?i:\b(?:password|passcode|pin|otp|passphrase)\s+(?:is|was|=)\s+)(?P<uval>\S+?)(?=[.,;!?]?(?:\s|$))"), None, "uval"),
    ("USERNAME", re.compile(r"(?i:\b(?:username|user\s*name|user\s*id|login|handle)\s+(?:is|was)\s+)@?(?P<uval>[\w.-]+)"), None, "uval"),
]

# Context triggers that introduce a person's name.  Strong triggers are enough
# on their own; weak ones ("I'm X") need the name to look unlike a common word.
_TITLE_NAME = r"(?P<name>[A-Z][a-z'’]+(?:[ -][A-Z][a-z'’]+){0,2})"
_STRONG_NAME_TRIGGERS = re.compile(
    r"(?:(?i:\bmy\s+name\s+is|\bname\s*:|\bnamed|\bcalled|\bsigned,?|\bregards,|\bsincerely,|\bpatient)\s+"
    r"|\b(?:Mr|Mrs|Ms|Mx|Miss|Dr|Prof|Sir|Madam|Smt|Shri)\.?\s+)" + _TITLE_NAME
)
_WEAK_NAME_TRIGGERS = re.compile(r"(?i:\b(?:i\s+am|i'm|this\s+is|it's|meet|my\s+(?:son|daughter|wife|husband|friend|mother|father|brother|sister|boss|colleague)))\s+" + _TITLE_NAME)
_LOCATION_TRIGGERS = re.compile(
    r"(?i:\b(?:i\s+live\s+in|i\s+am\s+from|i'm\s+from|live\s+in|lives\s+in|living\s+in|moved\s+to|based\s+in|located\s+in|resident\s+of|born\s+in|hometown\s+is))\s+"
    r"(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
)

# Words that are capitalised after a trigger but are not names.
_NOT_NAMES = {
    "a", "an", "the", "not", "very", "so", "just", "here", "there", "sorry", "fine",
    "good", "sure", "going", "having", "feeling", "trying", "looking", "working",
    "writing", "asking", "new", "unable", "please", "thanks", "hello", "hi", "happy",
    "sick", "ill", "tired", "worried", "confused", "interested", "done", "ready",
    "english", "indian", "american", "british", "diabetic", "pregnant",
    "allergic", "vegetarian", "vegan", "male", "female", "married", "single",
}

# NER false positives that are common in prompts (sentence-initial words, code).
_NER_STOPLIST = {
    "please", "what", "the", "this", "that", "how", "why", "when", "where", "which",
    "who", "can", "could", "would", "should", "will", "do", "does", "did", "hi",
    "hello", "hey", "thanks", "dear", "i", "we", "you", "my", "python", "java",
    "javascript", "sql", "api", "json", "http", "https", "linux", "windows",
    "docker", "git", "github", "fever", "covid", "doctor", "dr", "patient",
}


def _is_code_like(value: str) -> bool:
    """Identifiers such as ``ImportError``, ``psycopg2`` or ``os.path``."""
    if re.search(r"[_./\\()=<>{}\[\]]|\d", value):
        return True
    return bool(re.fullmatch(r"[A-Z]?[a-z]+(?:[A-Z][a-z0-9]*)+", value))


# ── NER backends ────────────────────────────────────────────────

_NER_CACHE = {}
_LABEL_TO_KIND = {"PER": "PERSON", "PERSON": "PERSON", "LOC": "LOCATION", "GPE": "LOCATION",
                  "ORG": "ORGANIZATION", "ORGANIZATION": "ORGANIZATION"}


def _transformer_entities(text: str, model_name: str, threshold: float, labels: Sequence[str], device: str) -> List[PIISpan]:
    key = (model_name, device)
    if key not in _NER_CACHE:
        from transformers import pipeline
        from .model_registry import resolve_device

        _NER_CACHE[key] = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="first",
            device=resolve_device(device),
        )
    ner = _NER_CACHE[key]
    spans = []
    for ent in ner(text):
        label = ent.get("entity_group", "")
        if label not in labels or float(ent["score"]) < threshold:
            continue
        spans.append(PIISpan(int(ent["start"]), int(ent["end"]), _LABEL_TO_KIND[label], float(ent["score"]), "ner"))
    return spans


def _nltk_entities(text: str) -> List[PIISpan]:
    try:
        import nltk
        from .nltk_resources import ensure

        ensure("tokenizers/punkt_tab", "taggers/averaged_perceptron_tagger_eng",
               "chunkers/maxent_ne_chunker_tab", "corpora/words")
        tokens = nltk.word_tokenize(text)
        tree = nltk.ne_chunk(nltk.pos_tag(tokens))
    except Exception:
        return []

    spans, cursor = [], 0
    for subtree in tree:
        leaves = subtree.leaves() if hasattr(subtree, "label") else [subtree]
        # Walk the source text to recover exact offsets (word_tokenize rewrites quotes).
        positions = []
        for word, _ in leaves:
            idx = text.find(word, cursor)
            if idx < 0:
                positions = []
                break
            positions.append((idx, idx + len(word)))
            cursor = idx + len(word)
        if not hasattr(subtree, "label") or not positions:
            continue
        kind = _LABEL_TO_KIND.get(subtree.label())
        if kind in ("PERSON", "LOCATION"):
            spans.append(PIISpan(positions[0][0], positions[-1][1], kind, 0.85, "ner"))
    return spans


# ── Public API ──────────────────────────────────────────────────

def _value_span(match: "re.Match[str]", group: Optional[str]) -> Tuple[int, int]:
    if group == "value":
        for name in ("qval", "uval"):
            if match.group(name) is not None:
                start, end = match.start(name), match.end(name)
                value = match.group(name)
                # Drop closing brackets that belong to surrounding code: f(password=abc)
                while value and value[-1] in ")]}" and value.count(value[-1]) > value.count({")": "(", "]": "[", "}": "{"}[value[-1]]):
                    value, end = value[:-1], end - 1
                return start, end
    if group:
        return match.start(group), match.end(group)
    return match.start(), match.end()


def _pattern_spans(text: str) -> List[PIISpan]:
    spans = []
    for kind, pattern, validator, group in _RECOGNIZERS:
        for match in pattern.finditer(text):
            if validator is not None and not validator(text, match):
                continue
            start, end = _value_span(match, group)
            if end > start:
                spans.append(PIISpan(start, end, kind))

    # Generic high-entropy tokens that no vendor-specific rule caught.
    for match in re.finditer(r"(?<![\w-])[A-Za-z0-9_\-+/]{20,}={0,2}(?![\w-])", text):
        if _looks_like_secret(match.group(0)):
            spans.append(PIISpan(match.start(), match.end(), "SECRET", 0.9, "entropy"))
    return spans


def _context_spans(text: str) -> List[PIISpan]:
    spans = []
    for match in _STRONG_NAME_TRIGGERS.finditer(text):
        name = match.group("name")
        if name.split()[0].lower() not in _NOT_NAMES:
            spans.append(PIISpan(match.start("name"), match.end("name"), "PERSON", 0.95, "context"))
    for match in _WEAK_NAME_TRIGGERS.finditer(text):
        name = match.group("name")
        first = name.split()[0].lower()
        if first not in _NOT_NAMES and first not in _NER_STOPLIST:
            spans.append(PIISpan(match.start("name"), match.end("name"), "PERSON", 0.8, "context"))
    for match in _LOCATION_TRIGGERS.finditer(text):
        spans.append(PIISpan(match.start("name"), match.end("name"), "LOCATION", 0.9, "context"))
    return spans


def _clean_entity(text: str, span: PIISpan) -> Optional[PIISpan]:
    start, end = span.start, span.end
    # Trim possessives and stray punctuation: "John's" -> "John"
    value = text[start:end]
    m = re.search(r"(?:['’]s|[.,;:!?'’\"])+$", value)
    if m:
        end -= len(m.group(0))
        value = text[start:end]
    # Titles are context, not identity: keep "Dr." and hide the name.
    m = re.match(r"(?:Mr|Mrs|Ms|Mx|Dr|Prof)\.?\s+", value)
    if m:
        start += m.end()
        value = text[start:end]
    if len(value) < 2 or value.lower() in _NER_STOPLIST:
        return None
    if span.source == "ner" and any(_is_code_like(tok) for tok in value.split()):
        return None
    # Reject spans that sit inside a larger identifier (e.g. part of a URL or code).
    if (start > 0 and re.match(r"[\w@]", text[start - 1])) or (end < len(text) and re.match(r"\w", text[end])):
        return None
    return PIISpan(start, end, span.kind, span.score, span.source)


def _propagate(text: str, spans: List[PIISpan]) -> List[PIISpan]:
    """Find every other mention of each detected named entity (and name parts)."""
    extra = []
    seen = set()
    for span in spans:
        if span.kind not in ("PERSON", "LOCATION", "ORGANIZATION"):
            continue
        value = text[span.start:span.end]
        variants = [value]
        if span.kind == "PERSON":
            variants += [p for p in value.split() if len(p) > 2]
        for variant in variants:
            if (span.kind, variant) in seen:
                continue
            seen.add((span.kind, variant))
            for m in re.finditer(rf"(?<![\w@.]){re.escape(variant)}(?![\w@])", text):
                extra.append(PIISpan(m.start(), m.end(), span.kind, span.score, "propagated"))
    return extra


def _resolve_overlaps(spans: List[PIISpan]) -> List[PIISpan]:
    ranked = sorted(spans, key=lambda s: (-_PRIORITY.get(s.kind, 0), -(s.end - s.start), s.start))
    accepted: List[PIISpan] = []
    for span in ranked:
        if all(span.end <= a.start or span.start >= a.end for a in accepted):
            accepted.append(span)
    return sorted(accepted, key=lambda s: s.start)


def find_pii(
    text: str,
    ner_backend: str = "none",
    ner_model: str = "dslim/distilbert-NER",
    ner_threshold: float = 0.8,
    ner_labels: Sequence[str] = ("PER", "LOC", "ORG"),
    propagate: bool = True,
    device: str = "auto",
) -> List[PIISpan]:
    """Return non-overlapping sensitive spans in source order."""
    if not text:
        return []

    spans = _pattern_spans(text)
    entity_spans = _context_spans(text)

    if ner_backend == "transformer":
        try:
            entity_spans += _transformer_entities(text, ner_model, ner_threshold, ner_labels, device)
        except Exception:
            entity_spans += _nltk_entities(text)
    elif ner_backend == "nltk":
        entity_spans += _nltk_entities(text)

    # Entities never override structured values (an email's local part is not a PERSON).
    structured = _resolve_overlaps(spans)
    cleaned = []
    for span in entity_spans:
        span = _clean_entity(text, span)
        if span and all(span.end <= s.start or span.start >= s.end for s in structured):
            cleaned.append(span)

    if propagate:
        cleaned += [
            s for s in _propagate(text, _resolve_overlaps(cleaned))
            if all(s.end <= t.start or s.start >= t.end for t in structured)
        ]

    return _resolve_overlaps(structured + cleaned)
