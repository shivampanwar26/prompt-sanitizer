"""High-precision detection of personally identifiable information and secrets.

Language-model surprise is not a reliable signal on its own: ordinary medical
words and uncommon phrasing are surprising too.  This module finds values
whose *form* or *context* is sufficient evidence that they are sensitive:

1. Validated pattern recognizers - a regex proposes, a checksum or parser
   disposes (Luhn for cards, mod-97 for IBAN, Verhoeff for Aadhaar,
   ``ipaddress`` for IPs, Shannon entropy for generic secrets).
2. Context recognizers - "my name is X", "Dr. X", "password is X",
   "patient ID: X" - which catch values no format rule can.
3. Named-entity recognition - a transformer NER model (PER / LOC / ORG),
   with NLTK's chunker as an offline fallback, filtered against code
   identifiers so ``ImportError`` or ``psycopg2`` are never "people".
4. Entity propagation - once "Sarah Connor" is found, every later "Sarah",
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
    "USERNAME": 76, "HOSTNAME": 75, "ADDRESS": 74, "DATE_TIME": 73, "PHONE": 70, "DATE": 65,
    "AGE": 64, "ID_NUMBER": 58, "POSTAL_CODE": 57, "PERSON": 50, "LOCATION": 45,
    "ORGANIZATION": 40, "TITLE": 60, "NRP": 44,
}


# Only explicit, delimited redactions qualify. Ordinary names and words such
# as "secret" can still be sensitive when they occur as field values.
_MASK_KINDS = sorted(set(_PRIORITY) | {
    "SENSITIVE", "REDACTED", "MASKED", "EMAIL_ADDRESS", "PHONE_NUMBER",
    "STREET_ADDRESS", "CREDIT_CARD", "NAME", "PII",
})
_MASK_LABEL = r"(?:" + "|".join(_MASK_KINDS) + r")(?:_\d+)?"
_MASK = re.compile(r"<" + _MASK_LABEL + r">|\[" + _MASK_LABEL + r"\]", re.I)


def masked_ranges(text: str) -> List[Tuple[int, int]]:
    """Offsets of explicit placeholders already present in the input."""
    ranges = [match.span() for match in _MASK.finditer(text)]
    if "***" in text or "\u2022" in text:
        for kind, pattern, _, group in _RECOGNIZERS:
            if kind not in ("PASSWORD", "SECRET"):
                continue
            for match in pattern.finditer(text):
                start, end = _value_span(match, group)
                if re.fullmatch(r"\*{3,}|\u2022{3,}", text[start:end]):
                    ranges.append((start, end))
    return sorted(set(ranges))


# ── Validators ──────────────────────────────────────────────────

def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)

# credit/debit card validation
def luhn_valid(value: str) -> bool:
    digits = _digits(value)
    if not 13 <= len(digits) <= 19:
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
    r"(?:phone|mobile|cell|call|tel|telephone|contact|whatsapp|text|sms|fax|number|no\.|message|answer(?:ing)?|reach|office)\W{0,3}(?:\w+\W{1,3}){0,3}$",
    re.I,
)


def _phone_ok(text: str, start: int, end: int) -> bool:
    value = text[start:end]
    digits = _digits(value)
    if not 7 <= len(digits) <= 15 or len(set(digits)) == 1:
        return False
    after = text[end:end + 24].lower()
    if re.match(r"\s*[-/]\s*(?:office|fax|mobile|cell|phone|tel)\b", after):
        return True
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
    r"|Court|Ct|Circle|Cir|Terrace|Ter|Trail|Trl|Parkway|Pkwy|Highway|Hwy|Marg|Nagar"
    r"|Katu|Tie|Tér|Rue|Strasse|Straße|Weg|Calle|Avenida|Via|Rua|Ulica"
    # Further standard street-type suffixes (USPS Publication 28 and UK usage).
    r"|Creek|Cliffs?|Coves?|Forges?|Crossing|Turnpike|Bypass|Mills?|Heights|Ridge|Close"
    r"|Crescent|Grove|Mews|Plaza|Glen|Harbou?r|Manor|Pike|Shores?|Summit|Vista|Estates?"
    r"|Expressway|Freeway|Junction|Causeway|Trace|Bluffs?|Canyon|Isle|Islands?|Meadows?"
    r"|Gardens|Points?|Passage|Hollow|Forks?|Ferry|Wharf|Quay|Loop|Walk|Row|Hills?|Springs?)"
)
_ADDRESS_STREET_NUMBER = re.compile(
    rf"\b(?:[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'-]*\s+){{1,3}}(?:(?:{_STREET_SUFFIXES}|u\.|str\.?)\s+)?\d{{1,5}}[A-Z]?\b",
    re.I,
)
_ADDRESS_CONTEXT_WORDS = {"a", "an", "and", "at", "address", "is", "new", "of", "to", "the"}
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
    # A card number is never embedded in an alphanumeric token ("U6292878...").
    ("CARD_NUMBER", re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])"),
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
    ("DATE_TIME", re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?\b"), None, None),
    ("DATE", re.compile(r"\bon\s+(?P<uval>Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.I), None, "uval"),
    ("AGE", re.compile(r"(?i:\bage(?:d)?(?:\s+is)?\s*[:=]?\s*)(?P<uval>\d{1,3})\b"),
     lambda t, m: 0 < int(m.group("uval")) <= 120, "uval"),
    ("AGE", re.compile(r"(?i:\b(?:age\s+of|retired\s+at\s+(?:the\s+)?age(?:\s+of)?|i['’]ll\s+(?:just\s+)?turn|(?:i|he|she|they|we)\s+(?:(?:just|recently|will)\s+)?turn(?:ed|ing)?|turn(?:ed|ing)?)\s*)(?P<uval>\d{1,3})\b"),
     lambda t, m: 0 < int(m.group("uval")) <= 120, "uval"),
    ("AGE", re.compile(r"(?i:\b(?:he|she|they|i)\s+was\s+)(?P<uval>\d{1,3})\b"),
     lambda t, m: 0 < int(m.group("uval")) <= 120, "uval"),
    ("AGE", re.compile(r"\b(?P<uval>\d{1,3})\s*(?:years?\s*old|yrs?\s*old|y/o|yo)\b", re.I),
     lambda t, m: 0 < int(m.group("uval")) <= 120, "uval"),
    ("PHONE", re.compile(
        r"(?<!\w)(?:\(\d{3}\)[ \t]*|\d{3}[-. ])\d{3}[-. ]\d{4}"
        r"[ \t]*(?:x|ext\.?)[ \t]*\d{1,6}\b", re.I), None, None),
    ("PHONE", re.compile(
        r"(?<![\w+.-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?)?(?:\d{7,15}|\d{2,6}(?:[\s.-]\d{2,6}){1,4})(?!\w)(?![.-]\d)"),
     lambda t, m: _phone_ok(t, m.start(), m.end()), None),
    # Military mail routes identify a delivery point across both address lines.
    ("ADDRESS", re.compile(
        r"\b(?:PSC|CMR|UNIT)[ \t]+\d{1,6},?[ \t]+BOX[ \t]+\d{1,6}"
        r"\s+(?:APO|FPO|DPO)[ \t]+(?:AA|AE|AP)[ \t]+\d{5}(?:-\d{4})?\b", re.I), None, None),
    # Ship-based military mail: "USNS Bergman / FPO AP 93757".
    ("ADDRESS", re.compile(
        r"\b(?:USNS|USNV|USS|USCGC)[ \t]+[^\W\d_][\w'’-]*(?:[ \t]+[^\W\d_][\w'’-]*)?"
        r"\s+(?:APO|FPO|DPO)[ \t]+(?:AA|AE|AP)[ \t]+\d{5}(?:-\d{4})?\b", re.I), None, None),
    # The abbreviated Polish street prefix precedes the name and building number.
    ("ADDRESS", re.compile(
        r"(?i:\bul\.)[ \t]+(?:[^\W\d_]+[ \t]+){1,3}\d{1,4}[A-Za-z]?\b"
    ), None, None),
    ("ADDRESS", re.compile(r"\b(?:P\.?[ \t]*O\.?[ \t]+Box|Post Office Box)[ \t]+[A-Z0-9-]*\d\b", re.I), None, None),
    ("ADDRESS", re.compile(rf"\b\d{{1,5}}(?:[A-Z])?,?\s+(?:[A-Z][a-z]+\s+){{1,3}}{_STREET_SUFFIXES}\b\.?"), None, None),
    ("ADDRESS", re.compile(r"\b(?:Flat|House|Apt|Apartment|Suite|Unit|Plot)\.?\s*(?:No\.?\s*)?#?\s*\d+[A-Z]?\b", re.I), None, None),
    ("POSTAL_CODE", re.compile(r"(?i:\b(?:zip|zip\s*code|postal\s*code|post\s*code|pin\s*code|pincode))(?:\s+(?:is|=))?\W{1,3}(?P<uval>[A-Z0-9]{3,4}\s?[A-Z0-9]{3}|\d{3,6}(?:\s+\d{2,3})?(?:-\d{3,4})?)\b"), None, "uval"),
    # City followed by a postal code, e.g. "in Bjert 6091". Requiring a
    # capitalized place immediately before 4-6 digits avoids masking bare years.
    ("POSTAL_CODE", re.compile(r"(?i:\b(?:in|at|from)\s+)(?:[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'-]*\s+){1,3}(?P<uval>\d{4,6})\b"), None, "uval"),
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

# Street-first conventions need local evidence to distinguish a house number
# from an arbitrary number following a person's or product's name.
_STREET_WORD = r"[^\W\d_]+(?:[-'\u2019][^\W\d_]+)*"
_BUILDING = r"\d{1,5}[A-Za-z]?\b"
_NAMED_STREET = re.compile(
    rf"(?i)\b(?:{_STREET_WORD}(?:straat|strasse|stra\u00dfe|gasse|vej|veien|stien|gata|braut|kuja|katu"
    r"|v\u00e4gen|vagen|gatan|gr\u00e4nd|stigen|gade|plads|polku|tie|weg|laan|dreef|plein|kade|gracht)"
    rf"|(?:rue|via|calle|avenida|avda\.|ul\.)[ \t]+(?:{_STREET_WORD}[ \t]+){{0,3}}{_STREET_WORD}"
    rf"|{_STREET_WORD}[ \t]+(?:strasse|stra\u00dfe|baan|terrasse|utca|all\u00e9e))"
    rf"[ \t]+{_BUILDING}"
)
_UNIT_LINE = re.compile(r"(?i)^[> \t]*(?:apt\.?|suite|unit|flat)[ \t]+(?:no\.?[ \t]*)?\w*\d")
_ADDRESS_BREAK = re.compile(
    r"(?i)^[> \t]*(?:[\w -]{1,32}[:=]|(?:please|then|send|call|deliver|leave|"
    r"return|use|keep|note|thanks|do not|I|we|you|they|this|that)\b)"
)
_ADDRESS_CONTEXT = re.compile(
    r"(?i)\b(?:(?:(?:home|mailing|billing|street|residential)\s+)?address"
    r"(?:\s+of\s+[^:\n]{1,60})?\s*(?:is\b|with\b|:|=)|(?:offices?|residence)\s+(?:is|are)\s+(?:located\s+)?at"
    r"|(?:send|sent|ship|shipped|deliver|delivered)\s+(?:it\s+)?to\??\s*(?:user:\s*)?"
    r"|(?:change|update|switch)\s+(?:my|the|our)\s+(?:(?:primary|secondary|billing|shipping|home)\s+(?:and\s+secondary\s+)?)?address\s+to\s+(?:the\s+same:\s*)?|lives?\s+at|(?:letter|parcel|package)\s+arrived\s+at(?=\s+\d))\s*"
)
_ADDRESS_FIELD = re.compile(
    r"(?i)^\s*(?:phone|mobile|telephone|email|e-mail|website|url|fax|notes?|contact)\s*[:=]"
)


def _trim_address_sentence(value: str) -> str:
    """Stop a captured address at a likely sentence boundary, preserving St. etc."""
    abbreviations = {"st", "rd", "dr", "ave", "blvd", "ln", "apt", "str", "no", "p.o", "u"}
    continuation = re.search(r"\s+(?=(?:after|before|call|please|then|when|where|while|for\s+(?:this\s+card|post(?:al)?\s+mail)|last\s+(?:night|week))\b)", value, re.I)
    if continuation:
        value = value[:continuation.start()].rstrip()
    for match in re.finditer(r"[.!?]\s+(?=[A-ZÀ-Þ])", value):
        prefix = value[:match.start()].rstrip()
        previous = prefix.rsplit(None, 1)[-1].rstrip(".,").lower() if prefix else ""
        if previous not in abbreviations:
            return value[:match.start()].rstrip()
    value = value.rstrip()
    if value.endswith(".") and value.split()[-1][:-1].casefold() not in abbreviations:
        return value[:-1].rstrip()
    return value


def _context_address_spans(text: str) -> List[PIISpan]:
    """Capture a labeled multi-line address as one span when it has address evidence."""
    spans = []
    for trigger in _ADDRESS_CONTEXT.finditer(text):
        start = trigger.end()
        tail = text[start:]
        lines = tail.splitlines(keepends=True)
        selected = []
        size = 0
        for line in lines[:6]:
            content = line.rstrip("\r\n")
            if not content.strip() or _ADDRESS_FIELD.match(content) or (selected and _ADDRESS_BREAK.match(content)):
                break
            if size + len(line) > 300:
                break
            selected.append(line)
            size += len(line)
        value = "".join(selected).rstrip()
        # Avoid swallowing a following sentence after a numbered address such
        # as "55 Main Street, Apt 4. Please call me tomorrow."
        value = _trim_address_sentence(value)
        if not value:
            continue
        # Require a street suffix, a unit marker, or a number before masking this block.
        evidence = (re.search(r"\d", value)
                    or re.search(rf"\b{_STREET_SUFFIXES}\b", value, re.I)
                    or re.search(r"(?i)\b(?:apt|apartment|suite|unit|flat|house)\b", value))
        if evidence:
            spans.append(PIISpan(start, start + len(value), "ADDRESS", 1.0, "address_context"))

    # International formats often put the street number after the street name
    # and omit an explicit "Address:" label. Require a nearby full postal number.
    line_starts = [0]
    for line in text.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    lines = text.splitlines()
    for line_index, line in enumerate(lines):
        line_start = line_starts[line_index]
        # International address records may place the postal code after separate
        # street, unit, city, and country lines. Keep the look-ahead bounded to
        # the same short record rather than requiring the code on the next line.
        context = "\n".join(lines[max(0, line_index - 1):line_index + 7])
        postal_evidence = bool(re.search(r"\b\d{5,6}\b", context))
        next_line = lines[line_index + 1] if line_index + 1 < len(lines) else ""
        has_unit_line = bool(_UNIT_LINE.match(next_line))
        numbered_record = re.match(r"^[> \t]*\d{1,6}\s+[^\W\d_]", line)
        named_streets = list(_NAMED_STREET.finditer(line))
        for street in named_streets:
            spans.append(PIISpan(line_start + street.start(), line_start + street.end(),
                                 "ADDRESS", 1.0, "street_number"))
        if not postal_evidence and not has_unit_line:
            continue
        named_street = named_streets[0] if named_streets else None
        number_street_match = re.search(
            rf"(?i)\b\d{{1,6}}[A-Z]?,?\s+(?:[\wÀ-ÖØ-öø-ÿ'-]+\s+){{1,4}}{_STREET_SUFFIXES}\b",
            line,
        )
        street_number_match = None
        for candidate in _ADDRESS_STREET_NUMBER.finditer(line):
            words_before_number = re.findall(r"[\wÀ-ÖØ-öø-ÿ'-]+", candidate.group(0))[:-1]
            if words_before_number and words_before_number[-1].casefold() not in _ADDRESS_CONTEXT_WORDS:
                street_number_match = candidate
                break
        # Prefer the suffix-anchored number-first form; the reverse form can
        # otherwise mistake prose such as "send a ... to 1987" for an address.
        street_match = number_street_match or named_street or street_number_match
        if street_match is None:
            continue
        if not (postal_evidence or numbered_record or named_street or number_street_match):
            continue

        # This is an address record: absorb its following city/country/postal
        # lines so they do not remain exposed after the street itself is masked.
        block_lines = []
        size = 0
        first_line = line[street_match.start():]
        prefix = re.match(r"^[> \t]*(?P<number>\d{1,6})[ \t]+", line)
        numeric_prefix = re.search(r"(?<![\w.-])(\d{1,6})[ \t]+$", line[:street_match.start()])
        if numeric_prefix and (street_match is named_street or street_match is number_street_match):
            address_offset = numeric_prefix.start(1)
            first_line = line[address_offset:]
        elif prefix and prefix.end() == street_match.start():
            address_offset = prefix.start("number")
            first_line = line[address_offset:]
        else:
            address_offset = street_match.start()
        for block_line in [first_line, *lines[line_index + 1:line_index + 7]]:
            if not block_line.strip() or _ADDRESS_FIELD.match(block_line) or (block_lines and _ADDRESS_BREAK.match(block_line)):
                break
            if size + len(block_line) > 360:
                break
            block_lines.append(block_line)
            size += len(block_line)
        if not block_lines:
            continue
        # Slice the original record so CRLF and LF produce identical offsets.
        last_line = line_index + len(block_lines) - 1
        value = text[line_start + address_offset:line_starts[last_line] + len(lines[last_line])].rstrip()
        value = _trim_address_sentence(value)
        if value:
            # Include a numeric postal prefix on the street line, when present.
            start = line_start + address_offset
            spans.append(PIISpan(start, start + len(value), "ADDRESS", 1.0, "address_block"))
    return spans

# Context triggers that introduce a person's name.  Strong triggers are enough
# on their own; weak ones ("I'm X") need the name to look unlike a common word.
_TITLE_NAME = r"(?P<name>[A-Z][a-z'’]+(?:[ -][A-Z][a-z'’]+){0,2})"
# A middle initial between two capitalized names is useful evidence even
# without a field label. Validate case separately to support Unicode names.
_NAME_WORD = r"[^\W\d_]+(?:['\u2019-][^\W\d_]+)*"
_DECLARED_NAME = re.compile(
    r"(?i:\b(?:my[ \t]+name[ \t]+is|(?:my|his|her)[ \t]+maiden[ \t]+name[ \t]+is|"
    r"people[ \t]+call[ \t]+me|full[ \t]+name[ \t]*:))[ \t]+"
    rf"(?P<name>{_NAME_WORD}(?:[ \t]+{_NAME_WORD}){{0,3}}?)"
    r"(?=[ \t]*(?:[.,;!?\r\n]|$)|[ \t]+(?i:and|but|from)\b)"
)
_INITIAL_NAME = re.compile(
    r"(?<!\w)([^\W\d_]{2,})[ \t]+([^\W\d_])\.[ \t]+([^\W\d_]{2,})(?!\w)"
)
_STRONG_NAME_TRIGGERS = re.compile(
    r"(?:(?i:\bmy\s+name\s+is|\bname\s*:|\bnamed|\bcalled|\bsigned,?|\bregards,|\bsincerely,|\bpatient)\s+"
    r"|\b(?:Mr|Mrs|Ms|Mx|Miss|Dr|Prof|Sir|Madam|Smt|Shri)\.?\s+)" + _TITLE_NAME
)
_WEAK_NAME_TRIGGERS = re.compile(r"(?i:\b(?:i\s+am|i'm|this\s+is|it's|meet|my\s+(?:son|daughter|wife|husband|friend|mother|father|brother|sister|boss|colleague)))\s+" + _TITLE_NAME)
_LOCATION_TRIGGERS = re.compile(
    r"(?i:\b(?:originally\s+from|i\s+live\s+in|i\s+am\s+from|i'm\s+from|live\s+in|lives\s+in|living\s+in|moved\s+to|moved\s+(?:here\s+)?from|based\s+in|located\s+in|resident\s+of|born\s+in|hometown\s+is))\s+"
    r"(?P<name>[\wÀ-ÖØ-öø-ÿ'’-]+(?:[ \t]+[\wÀ-ÖØ-öø-ÿ'’-]+){0,2}(?:[ \t]*\([^()\n]{1,20}\))?)"
)
_LOCATION_FIELD = re.compile(
    r"(?im)^\s*where\s*:\s*(?P<name>[\wÀ-ÖØ-öø-ÿ'’-]+(?:[ \t]+[\wÀ-ÖØ-öø-ÿ'’-]+){0,2})"
    r"(?=\s+(?:country|county|state|province)\b|[.,;\n]|$)"
)

# Words that are capitalised after a trigger but are not names.
_NOT_NAMES = {
    "a", "an", "the", "not", "very", "so", "just", "here", "there", "sorry", "fine",
    "good", "sure", "going", "having", "feeling", "trying", "looking", "working",
    "writing", "asking", "new", "unable", "please", "thanks", "hello", "hi", "happy",
    "sick", "ill", "tired", "worried", "confused", "interested", "done", "ready",
    "english", "indian", "american", "british", "diabetic", "pregnant",
    "allergic", "vegetarian", "vegan", "male", "female", "married", "single",
    # "meet Tuesday" / "I'm June-born": weekdays and months are not names.
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july", "august", "september",
    "october", "november", "december", "today", "tomorrow", "tonight",
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
    # Surname prefixes produce camel-case names (McDowell, MacLeod, DeVito,
    # LaFontaine) that are people, not identifiers.
    if re.fullmatch(r"(?:Mc|Mac|De|Di|Da|Du|La|Le|Van|Von|O')[A-Z][a-z]+", value):
        return False
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
        start, end = int(ent["start"]), int(ent["end"])
        if _LABEL_TO_KIND[label] == "PERSON":
            # The token classifier may omit a middle initial or surname.
            # Extend only across adjacent name-shaped tokens.
            # A name continues on the same line only: "[ \t]" not "\s", so a
            # signature's name never swallows the city line beneath it.
            suffix = re.match(
                r"(?:[ \t]+(?:(?:[A-ZÀ-ÖØ-Þ]\.)[ \t]+)?"
                r"(?:(?:de|del|der|la|ten|van|von)[ \t]+)?"
                r"[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’-]*){1,2}",
                text[end:],
            )
            if suffix:
                end += suffix.end()
        spans.append(PIISpan(start, end, _LABEL_TO_KIND[label], float(ent["score"]), "ner"))
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
    spans.extend(_context_address_spans(text))
    spans.extend(_structural_spans(text))
    return spans


# Titles and workplace affiliations can identify someone in a profile even
# when neither the occupation nor the employer is sensitive in ordinary prose.
_HONORIFIC = re.compile(
    r"\b(?i:mr|mrs|ms|mx|dr|prof)\.(?=[ \t]+[^\W\d_]{2,}\b)"
)
_ROLE_END = (
    r"(?:administrator|analyst|assistant|clerk|consultant|controller|coordinator|"
    r"designer|developer|director|driver|engineer|librarian|manager|mechanic|nurse|"
    r"officer|operator|scientist|specialist|supervisor|teacher|technician|writer|"
    r"accountant|agent|architect|assembler|attendant|attorney|auditor|baker|banker|"
    r"broker|carpenter|cashier|chef|chemist|cleaner|counsel+or|courier|dentist|"
    r"dispatcher|distributor|economist|editor|electrician|estimator|fabricator|"
    r"farmer|firefighter|gardener|inspector|installer|instructor|interpreter|"
    r"investigator|janitor|journalist|lawyer|lecturer|machinist|marketer|"
    r"paralegal|paramedic|pharmacist|photographer|physician|physicist|pilot|planner|"
    r"plumber|programmer|receptionist|recruiter|repairer|representative|researcher|"
    r"secretary|soldier|statistician|strategist|surgeon|therapist|translator|tutor|"
    r"underwriter|welder|worker)"
)
_ROLE = r"(?:[^\W\d_]+(?:['?-][^\W\d_]+)*,?[ \t]+){0,6}" + _ROLE_END
_ROLE_FIELD = re.compile(
    r"(?im)^[ \t]*(?:(?:job title|occupation|profession)[ \t]*[:=][ \t]*(?P<field>"
    + _ROLE + r")\b|(?P<profile>" + _ROLE
    + r")[ \t]*\r?\n(?:[ \t]*\r?\n)?[ \t]*(?:personal info(?:rmation)?|contact details)[ \t]*:)"
)
_EMPLOYER = re.compile(
    r"(?i:\b(?:work(?:ed|s)?[ \t]+for[ \t]+|employer[ \t]*[:=][ \t]*))"
    r"(?P<name>[\w&'-]+(?:[ \t]+[\w&'-]+){0,5}?)(?=[ \t]+(?i:as|since|until|where)\b|[ \t]*[.,;!?\r\n]|[ \t]*$)"
)


# ── Structural recognizers ──────────────────────────────────────
# These capture multi-word entities as ONE span so that the connecting words
# ("and", "the corner of") are replaced together with the names they join;
# otherwise per-name detection leaves "X, Y and Z" half exposed.

# Capitalised words that make "A, B and C" a list rather than a firm name.
_LIST_NOT_FIRM = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "english", "french", "spanish",
    "german", "italian", "chinese", "japanese", "hindi", "arabic", "russian",
    "portuguese", "dutch", "red", "green", "blue", "black", "white", "yellow",
    "python", "java", "javascript", "c", "go", "rust", "ruby", "north", "south",
    "east", "west", "mr", "mrs", "ms", "dr",
}
_FIRM_WORD = r"[A-ZÀ-Þ][\wÀ-ÿ'’]+"
_FIRM_LIST = re.compile(rf"\b{_FIRM_WORD},[ \t]+{_FIRM_WORD}[ \t]+(?:and|&)[ \t]+{_FIRM_WORD}\b")
# Lower-case firm names are only trusted on a line of their own (signature blocks).
_FIRM_LIST_LINE = re.compile(
    r"(?m)^[>?\s]*(?P<name>[a-z][\w'’]+,[ \t]+[a-z][\w'’]+[ \t]+and[ \t]+[a-z][\w'’]+)[ \t]*$"
)
_COMPANY_SUFFIX = re.compile(
    r"(?i)(?<![\w-])(?P<name>(?:[\w&'’-]+[ \t]+){1,3})(?:inc|llc|ltd|corp|corporation|gmbh|plc)\b\.?"
)
_COMPANY_LEAD_STOP = {
    "from", "at", "is", "was", "of", "for", "with", "to", "in", "the", "a", "an", "by",
    "and", "or", "joined", "join", "work", "works", "worked", "on", "my", "our", "your",
    "his", "her", "their", "spent", "year", "years",
}
_HYPHEN_FIRM = re.compile(r"(?<![\w-])(?P<a>[A-Z][a-z]{2,})-(?P<b>[A-Z][a-z]{2,})(?![\w-])")
_HYPHEN_FIRM_LINE = re.compile(r"(?m)^[>?\s]*(?P<name>[a-z]{3,}-[a-z]{3,})[ \t]*$")
_CORNER = re.compile(
    # {braces} are template variables ("{street_name}"), never real values.
    # Periods are allowed inside the street parts for abbreviations ("ul.", "Str.").
    r"(?i)\bthe[ \t]+corner[ \t]+of[ \t]+(?P<a>[^,;!?\n{}]{2,60}?)[ \t]+and[ \t]+"
    r"(?P<b>[^\s,;!?{}]+(?:[ \t]+[^\s,;!?{}]+){0,3}?)\.?"
    r"(?=[ \t]*(?:[,.;!?\n]|$)|[ \t]+(?:in|on|near|by|at|for|and|after|before)\b)"
)

# Nationality, religious and political affiliation stated about oneself or a
# group ("I am cambodian", "we are proud saudis"). Adjectives used about
# things ("Russian government", "French film") are not personal data and are
# deliberately left alone.
_NRP_WORDS = {
    "afghan", "albanian", "algerian", "american", "andorran", "angolan", "argentine",
    "argentinian", "armenian", "australian", "austrian", "azerbaijani", "bahamian",
    "bahraini", "bangladeshi", "barbadian", "belarusian", "belgian", "belizean",
    "beninese", "bhutanese", "bolivian", "bosnian", "botswanan", "brazilian", "british",
    "bruneian", "bulgarian", "burkinabe", "burmese", "burundian", "cambodian",
    "cameroonian", "canadian", "chadian", "chilean", "chinese", "colombian",
    "congolese", "croatian", "cuban", "cypriot", "czech", "danish", "djiboutian",
    "dominican", "dutch", "ecuadorean", "ecuadorian", "egyptian", "emirati", "english",
    "eritrean", "estonian", "ethiopian", "fijian", "filipino", "finnish", "french",
    "gabonese", "gambian", "georgian", "german", "ghanaian", "greek", "guatemalan",
    "guinean", "guyanese", "haitian", "honduran", "hungarian", "icelandic", "indian",
    "indonesian", "iranian", "iraqi", "irish", "israeli", "italian", "ivorian",
    "jamaican", "japanese", "jordanian", "kazakh", "kenyan", "korean", "kosovan",
    "kuwaiti", "kyrgyz", "lao", "latvian", "lebanese", "liberian", "libyan",
    "lithuanian", "luxembourgish", "macedonian", "malagasy", "malawian", "malaysian",
    "maldivian", "malian", "maltese", "mauritanian", "mauritian", "mexican",
    "moldovan", "mongolian", "montenegrin", "moroccan", "mozambican", "namibian",
    "nepalese", "nepali", "nicaraguan", "nigerian", "nigerien", "norwegian", "omani",
    "pakistani", "palestinian", "panamanian", "paraguayan", "peruvian", "polish",
    "portuguese", "qatari", "romanian", "russian", "rwandan", "salvadoran", "samoan",
    "saudi", "scottish", "senegalese", "serbian", "singaporean", "slovak", "slovakian",
    "slovenian", "somali", "spanish", "sudanese", "surinamese", "swedish", "swiss",
    "syrian", "taiwanese", "tajik", "tanzanian", "thai", "togolese", "tongan",
    "trinidadian", "tunisian", "turkish", "turkmen", "ugandan", "ukrainian",
    "uruguayan", "uzbek", "venezuelan", "vietnamese", "welsh", "yemeni", "zambian",
    "zimbabwean",
    # religious and political affiliation
    "christian", "catholic", "protestant", "orthodox", "muslim", "sunni", "shia",
    "shii", "shiite", "hindu", "buddhist", "jewish", "jew", "sikh", "jain", "atheist",
    "agnostic", "mormon", "democrat", "republican", "socialist", "communist",
    "conservative", "liberal", "libertarian",
}
_NRP_SELF = re.compile(
    r"(?i)\b(?:i[ \t]+am|i'm|we[ \t]+are|we're|he[ \t]+is|she[ \t]+is|they[ \t]+are)"
    r"[ \t]+(?:(?:a|an|also|proud|both|all|devout|practicing|practising)[ \t]+){0,2}"
    r"(?P<nrp>[^\W\d_]+)\b"
)


def _nrp_word(word: str) -> bool:
    w = word.casefold()
    return w in _NRP_WORDS or (w.endswith("s") and (w[:-1] in _NRP_WORDS or w[:-2] in _NRP_WORDS))


_ROLE_AS = re.compile(
    r"(?i:\bwork(?:ed|s|ing)?[ \t]+(?:for|at)[ \t]+[^,.\n]{1,60}?[ \t]+as[ \t]+(?:an?[ \t]+)?)"
    r"(?P<role>" + _ROLE + r")\b",
)
_ROLE_APPOSITIVE = re.compile(r"(?i),[ \t]+the[ \t]+(?P<role>" + _ROLE + r"),")
_ROLE_ONLY_LINE = re.compile(r"(?i)^[>?\s]*(?P<role>" + _ROLE + r")[ \t]*$")


def _role_line_spans(text: str) -> List[PIISpan]:
    """A line that is only a job title, directly under a short name line (signature block)."""
    spans = []
    offset = 0
    previous = ""
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        match = _ROLE_ONLY_LINE.match(content)
        prev_words = re.findall(r"[^\W\d_]+", previous)
        if match and 1 <= len(prev_words) <= 4 and not re.search(r"[:@\d]", previous):
            spans.append(PIISpan(offset + match.start("role"), offset + match.end("role"),
                                 "TITLE", 0.9, "context"))
        if content.strip():
            previous = content
        offset += len(line)
    return spans


# Street names introduced by a prefix (Romance/Iberian conventions), with the
# house number before ("33 avenue de Provence") or after ("Rua Mina 50").
_STREET_PREFIX = r"(?:rue|rua|av\.?|avenue|avenida|avda\.|via|viale|calle|c/|strada|largo|praça|piazza|rambla|travessa|camino)"
_PARTICLES = r"(?:(?:de|du|des|la|le|les|del|da|do|dos|das|di|della|von|van|der)[ \t]+)*"
_PREFIX_WORD = r"[^\W\d_][\w'’-]*"
_PREFIX_STREET_NUMBER_FIRST = re.compile(
    rf"(?i)(?<![\w-])\d{{1,5}}[A-Za-z]?[ \t]+{_STREET_PREFIX}[ \t]+{_PARTICLES}{_PREFIX_WORD}"
    rf"(?:[ \t]+{_PARTICLES}{_PREFIX_WORD}){{0,3}}"
)
_PREFIX_STREET_NUMBER_AFTER = re.compile(
    rf"(?i)(?<![\w-]){_STREET_PREFIX}[ \t]+{_PARTICLES}{_PREFIX_WORD}"
    rf"(?:[ \t]+{_PARTICLES}{_PREFIX_WORD}){{0,3}}[ \t]+\d{{1,5}}[A-Za-z]?\b"
)
# Central-European street words follow the name: "Árpád fejedelem útja 3.", "hegyalja út 98."
_SUFFIX_STREET = re.compile(
    rf"(?i)(?<![\w-])(?:{_PREFIX_WORD}[ \t]+){{1,3}}(?:út|útja|utca|u\.|tér|körút|rkp\.|nábřeží|náměstí)"
    r"[ \t]+\d{1,5}\.?(?!\d)"
)
# "lives on Hoge Wei 32 street", "enter on Rua Mina 50 St."
_ON_STREET = re.compile(
    r"(?i)\b(?:lives?|living|located|enter|entrance\s+is|situated)\s+on\s+"
    r"(?P<street>[^\s,.;!?]+(?:[ \t]+[^\s,.;!?]+){0,4}?)\.?[ \t]+(?:street|st)\b"
)
# A line holding only "Firstname [M.] Lastname" in a multi-line record (a
# signature or profile header). The first name must be in the names gazetteer.
_NAME_LINE = re.compile(
    r"(?m)^[>?\s]*(?P<name>[^\W\d_]+(?:[ \t]+[^\W\d_]\.?)?(?:[ \t]+[^\W\d_][\w'’-]*){1,2})[ \t]*$"
)


_TRIM_WORDS = {"street", "st", "near", "next", "opposite", "behind", "then", "after", "before"}


def _trim_function_words(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    """Shrink [start, end) so it neither starts nor ends with a function word,
    and cut it at the first function word in the middle ("rue Royale and it")."""
    tokens = list(re.finditer(r"\S+", text[start:end]))
    stop = _COMMON_LOWER | _TRIM_WORDS

    def is_stop(tok):
        return tok.group().casefold().rstrip(".,") in stop and tok.group().casefold() != "nan"

    while tokens and is_stop(tokens[0]):
        tokens.pop(0)
    for i, tok in enumerate(tokens):
        if i and is_stop(tok):
            tokens = tokens[:i]
            break
    if not tokens or not re.search(r"[^\W\d_]", text[start + tokens[0].start():start + tokens[-1].end()]):
        return None
    return start + tokens[0].start(), start + tokens[-1].end()


def _name_line_spans(text: str) -> List[PIISpan]:
    if text.count("\n") < 2:
        return []
    names = _first_names()
    spans = []
    for match in _NAME_LINE.finditer(text):
        words = match.group("name").split()
        if words[0].casefold() not in names:
            continue
        lowered = [w.casefold().rstrip(".") for w in words]
        if any(w in _COMMON_LOWER for w in lowered) or re.search(_ROLE_END, match.group("name"), re.I):
            continue
        if not (all(w[:1].isupper() for w in words) or all(w.islower() for w in words)):
            continue
        spans.append(PIISpan(*match.span("name"), "PERSON", 0.9, "name_line"))
    return spans


def _structural_spans(text: str) -> List[PIISpan]:
    from .nltk_resources import is_dictionary_word

    spans = []
    for pattern in (_PREFIX_STREET_NUMBER_FIRST, _PREFIX_STREET_NUMBER_AFTER, _SUFFIX_STREET):
        for match in pattern.finditer(text):
            trimmed = _trim_function_words(text, match.start(), match.end())
            if trimmed:
                spans.append(PIISpan(*trimmed, "ADDRESS", 0.9, "street_prefix"))
    for match in _ON_STREET.finditer(text):
        spans.append(PIISpan(*match.span("street"), "ADDRESS", 0.9, "street_context"))
    spans.extend(_name_line_spans(text))
    for match in _FIRM_LIST.finditer(text):
        words = re.findall(r"[^\W\d_]+", match.group(0))
        if not any(w.casefold() in _LIST_NOT_FIRM for w in words if w.casefold() not in {"and"}):
            spans.append(PIISpan(*match.span(), "ORGANIZATION", 0.9, "org_pattern"))
    for match in _FIRM_LIST_LINE.finditer(text):
        words = re.findall(r"[^\W\d_]+", match.group("name"))
        if not any(w.casefold() in _LIST_NOT_FIRM for w in words if w != "and"):
            spans.append(PIISpan(*match.span("name"), "ORGANIZATION", 0.85, "org_pattern"))
    for match in _COMPANY_SUFFIX.finditer(text):
        start = match.start("name")
        words = list(re.finditer(r"[\w&'’-]+", match.group("name")))
        while words and words[0].group().casefold() in _COMPANY_LEAD_STOP:
            words.pop(0)
        if words:
            spans.append(PIISpan(start + words[0].start(), match.end(), "ORGANIZATION", 0.9, "org_pattern"))
    for match in _HYPHEN_FIRM.finditer(text):
        a, b = match.group("a"), match.group("b")
        if not (is_dictionary_word(a) and is_dictionary_word(b)):
            spans.append(PIISpan(*match.span(), "ORGANIZATION", 0.8, "org_pattern"))
    for match in _HYPHEN_FIRM_LINE.finditer(text):
        a, b = match.group("name").split("-", 1)
        if not (is_dictionary_word(a) and is_dictionary_word(b)):
            spans.append(PIISpan(*match.span("name"), "ORGANIZATION", 0.8, "org_pattern"))
    for match in _CORNER.finditer(text):
        spans.append(PIISpan(match.start(), match.end("b"), "ADDRESS", 0.95, "address_corner"))
    for match in _NRP_SELF.finditer(text):
        if _nrp_word(match.group("nrp")):
            spans.append(PIISpan(*match.span("nrp"), "NRP", 0.9, "context"))
    for pattern in (_ROLE_AS, _ROLE_APPOSITIVE):
        for match in pattern.finditer(text):
            spans.append(PIISpan(*match.span("role"), "TITLE", 0.9, "context"))
    spans.extend(_role_line_spans(text))
    return spans


def _context_spans(text: str) -> List[PIISpan]:
    spans = []
    for match in _DECLARED_NAME.finditer(text):
        words = match.group("name").casefold().split()
        if not any(word in _NOT_NAMES or word in {"unknown", "anonymous", "redacted", "masked", "private", "available", "withheld"} for word in words):
            spans.append(PIISpan(*match.span("name"), "PERSON", 0.95, "context"))
    for match in _HONORIFIC.finditer(text):
        spans.append(PIISpan(*match.span(), "TITLE", 0.95, "context"))
    for match in _ROLE_FIELD.finditer(text):
        spans.append(PIISpan(*match.span(match.lastgroup), "TITLE", 0.95, "context"))
    for match in _EMPLOYER.finditer(text):
        value = match.group("name")
        if any(ch.isupper() for ch in value) and value.split()[0].lower() not in {
            "a", "an", "the", "my", "this", "that", "me", "us", "you",
        }:
            spans.append(PIISpan(*match.span("name"), "ORGANIZATION", 0.95, "context"))
    for match in _INITIAL_NAME.finditer(text):
        first, initial, last = match.groups()
        if first.istitle() and initial.isupper() and last[0].isupper() and not last.isupper():
            spans.append(PIISpan(*match.span(), "PERSON", 0.95, "context"))
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
        # A location trigger can be followed immediately by a new clause.
        # Preserve compound place names such as Trinidad and Tobago
        # when there is no evidence of a following verb.
        name = match.group("name")
        boundary = re.search(r"\s+(?:and|but|then)\b", name)
        if boundary and (not name[boundary.end():].strip()
                         or re.match(r"[ \t]+(?:works?|lives?|said|is|was|has|will)\b",
                                     text[match.end("name"):], re.I)
                         or name[boundary.end():].lstrip()[:1].islower()):
            name = name[:boundary.start()]
        spans.append(PIISpan(match.start("name"), match.start("name") + len(name),
                             "LOCATION", 0.9, "context"))
    for match in _LOCATION_FIELD.finditer(text):
        spans.append(PIISpan(match.start("name"), match.end("name"), "LOCATION", 0.95, "context"))
    return spans


def _clean_entity(text: str, span: PIISpan) -> Optional[PIISpan]:
    start, end = span.start, span.end
    # Trim possessives and stray punctuation: "John's" -> "John"
    value = text[start:end]
    m = re.search(r"(?:['’]s|[.,;:!?'’\"])+$", value)
    if m and span.kind != "TITLE":
        end -= len(m.group(0))
        value = text[start:end]
    # Titles are context, not identity: keep "Dr." and hide the name.
    m = re.match(r"(?:Mr|Mrs|Ms|Mx|Dr|Prof)\.?\s+", value)
    if m:
        start += m.end()
        value = text[start:end]
    if len(value) < 2 or value.lower() in _NER_STOPLIST:
        return None
    if span.source.startswith("ner") and any(_is_code_like(tok) for tok in value.split()):
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


# ── Truecasing for a case-sensitive NER model ───────────────────
# The token classifier relies heavily on capitalisation, so it misses
# lower-cased names ("selma is from ...") and ALL-CAPS places ("STAVANGER").
# A second pass over a length-preserving truecased copy recovers them; offsets
# map back 1:1 because no character changes length.

_COMMON_LOWER = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by",
    "for", "with", "from", "as", "is", "are", "was", "were", "be", "been", "it",
    "this", "that", "i", "me", "my", "you", "your", "he", "she", "his", "her", "we",
    "our", "they", "their", "what", "which", "who", "how", "why", "when", "where",
    "not", "no", "yes", "do", "does", "did", "have", "has", "had", "can", "could",
    "should", "would", "will", "please", "also", "very", "nan",
}
_TRUECASE_WORD = re.compile(r"(?<![\w@./-])[^\W\d_][^\W\d_'’-]*(?![\w@/])")


def _first_names():
    """NLTK's names corpus (a general first-name gazetteer), or an empty set offline."""
    global _FIRST_NAME_SET
    if _FIRST_NAME_SET is None:
        _FIRST_NAME_SET = set()
        try:
            from .nltk_resources import ensure

            if ensure("corpora/names"):
                from nltk.corpus import names

                _FIRST_NAME_SET = {n.casefold() for n in names.words()}
        except Exception:
            pass
    return _FIRST_NAME_SET


_FIRST_NAME_SET = None


def _truecase(text: str) -> str:
    """Capitalise likely proper nouns that are written lower-case or ALL-CAPS.

    A lower-case word is capitalised when it is a known first name or is not an
    English dictionary word (and is not a common function word). An ALL-CAPS
    word of 4+ letters is title-cased. Every change preserves length.
    """
    from .nltk_resources import is_dictionary_word

    names = _first_names()
    chars = list(text)
    for match in _TRUECASE_WORD.finditer(text):
        word = match.group(0)
        if word.islower():
            lower = word.casefold()
            if lower in _COMMON_LOWER or len(word) < 3:
                continue
            if lower in names or not is_dictionary_word(word):
                head = word[0].upper()
                if len(head) == 1:
                    chars[match.start()] = head
        elif word.isupper() and len(word) >= 4:
            cased = word[0] + word[1:].lower()
            if len(cased) == len(word):
                chars[match.start():match.end()] = list(cased)
    return "".join(chars)


# ── Address consolidation ───────────────────────────────────────
# Detectors often find an address in fragments ("Kuuse 53 Apt. 641",
# "Grubbenvorst", "Netherlands") and miss the house number in front of a
# street they recognise as a location. Masking fragments separately leaves
# the number, connecting tokens and trailing city/postal code exposed.

_STREETISH = re.compile(
    r"(?i)(?:^|\s)(?:av|ave|avenue|avenida|rua|rue|ul|via|calle|c/|str|strasse|straße|street|st|"
    r"road|rd|utca|út|útja|gasse|vej|veien|katu|tie|gatan|vägen|weg|laan|straat|lane|drive|"
    r"way|place|plaza|platz|square|boulevard|blvd|close|crescent|court|terrace)\.?(?:\s|$)"
)
_ADDRESS_FAMILY = {"ADDRESS", "LOCATION", "POSTAL_CODE"}
_UNIT_WORDS = {"apt", "apt.", "apartment", "suite", "unit", "box", "flat", "floor", "fl", "fl."}


def _absorb_address_tail(text: str, end: int) -> int:
    """Extend an address over the rest of its line when that rest is still address-like
    (unit, city in capitals, region code, postal number); stop at prose."""
    line_end = text.find("\n", end)
    line_end = len(text) if line_end < 0 else line_end
    tail = _trim_address_sentence(text[end:line_end])
    pos, new_end, evidence = 0, end, False
    for token in re.finditer(r"[^\s,()]+", tail):
        gap = tail[pos:token.start()]
        if gap.strip(" \t,()"):
            break
        word = token.group(0)
        bare = word.rstrip(".")
        if not bare:            # the "." closing an abbreviation such as "Apt."
            pos = token.end()
            continue
        if bare.casefold() in _COMMON_LOWER and bare.casefold() != "nan":
            break
        if any(ch.isdigit() for ch in bare):
            evidence = True
        elif bare.isupper() and len(bare) >= 2:
            evidence = True
        elif bare.casefold() in _UNIT_WORDS:
            pass
        elif bare[:1].isupper():
            pass
        elif len(bare) <= 3 and bare.isalpha():
            pass
        else:
            break
        pos = token.end()
        if evidence:
            new_end = end + pos
    return new_end


def _consolidate_addresses(text: str, spans: List[PIISpan]) -> List[PIISpan]:
    spans = sorted(spans, key=lambda s: s.start)
    taken = [(s.start, s.end) for s in spans]

    def free(start: int, end: int, own: PIISpan) -> bool:
        return all(end <= a or start >= b for a, b in taken if (a, b) != (own.start, own.end))

    # 1. House number(s) directly in front of a street.
    widened = []
    for span in spans:
        # A name between a house number and a building number is a street,
        # whatever the NER label: "641 Dašická 688".
        if span.kind in ("ORGANIZATION", "PERSON", "LOCATION"):
            sandwich_before = re.search(r"(?:^|(?<=[\s,:>]))(\d{1,6}[A-Za-z]?)[ \t]+$", text[:span.start])
            sandwich_after = re.match(r"[ \t]+\d{1,5}[A-Za-z]?\b", text[span.end:])
            if sandwich_before and sandwich_after:
                start, end = sandwich_before.start(1), span.end + sandwich_after.end()
                if free(start, end, span):
                    span = PIISpan(start, end, "ADDRESS", span.score, "address_sandwich")
                    widened.append(span)
                    continue
        if span.kind in ("ADDRESS", "LOCATION"):
            line_start = text.rfind("\n", 0, span.start) + 1
            before = text[line_start:span.start]
            number = re.search(r"(?:^|(?<=[\s,:>]))((?:\d{1,6}[A-Za-z]?[ \t]+){1,2})$", before)
            value = text[span.start:span.end]
            after = text[span.end:span.end + 8]
            streetlike = span.kind == "ADDRESS" or _STREETISH.search(" " + value + " ") \
                or re.match(r"[ \t]+\d{1,5}[A-Za-z]?\b", after)
            if number and streetlike:
                start = line_start + number.start(1)
                end = span.end
                trailing = re.match(r"[ \t]+\d{1,5}[A-Za-z]?\b\.?", after)
                if span.kind == "LOCATION" and trailing:
                    end += trailing.end()
                if free(start, end, span):
                    span = PIISpan(start, end, "ADDRESS", span.score, "address_number")
        widened.append(span)

    # 2. Merge adjacent fragments of one address (same line, or the next line).
    merged: List[PIISpan] = []
    for span in widened:
        prev = merged[-1] if merged else None
        if (prev is not None and prev.kind in _ADDRESS_FAMILY and span.kind in _ADDRESS_FAMILY
                and "ADDRESS" in (prev.kind, span.kind)
                and re.fullmatch(r"[ \t,]*(?:\r?\n[ \t>?]*)?[ \t,]*", text[prev.end:span.start])):
            merged[-1] = PIISpan(prev.start, span.end, "ADDRESS", max(prev.score, span.score), "address_merge")
        else:
            merged.append(span)

    # 3. Absorb the address-like remainder of the line (city, region, postal
    #    code), then any following city / ", REGION" / postal / country lines.
    result = []
    for i, span in enumerate(merged):
        if span.kind == "ADDRESS":
            limit = merged[i + 1].start if i + 1 < len(merged) else len(text)
            end = min(_absorb_address_tail(text, span.end), limit)
            end = min(_absorb_address_lines(text, end), limit)
            if end > span.end:
                span = PIISpan(span.start, end, "ADDRESS", span.score, span.source)
        result.append(span)
    return result


# A trailing address line: a city in capitals or title case, ", VIC", a postal
# number, or "Country 12345" -- short, no colon (which would be a field label).
_ADDRESS_LINE = re.compile(
    r"^[>?\s]*,?[ \t]*(?:[^\W\d_][\w'’.()/-]*[ \t]*){0,4}(?:\d{3,6}(?:[ \t-]\d{2,4})?)?[ \t]*\.?$"
)


def _absorb_address_lines(text: str, end: int) -> int:
    """Extend over up to 4 following lines that continue the same postal record.
    Only lines up to the last one with evidence (capitals or a postal number) count."""
    if text[end:end + 1] not in ("\r", "\n") and text[end:].split("\n", 1)[0].strip():
        return end
    pos, best = end, end
    for _ in range(4):
        nl = text.find("\n", pos)
        if nl < 0:
            break
        line_end = text.find("\n", nl + 1)
        line_end = len(text) if line_end < 0 else line_end
        line = text[nl + 1:line_end].rstrip("\r")
        content = line.strip(" \t>?")
        if not content or len(content) > 45 or ":" in content or not _ADDRESS_LINE.match(line):
            # A postal code opening a line of prose still closes the record:
            # "\n 77233 for this card: ..." -> absorb just "77233".
            postal = re.match(r"^[>?\s]*,?[ \t]*\d{3,6}(?:[ \t-]\d{2,4})?(?![\w-])", line)
            if postal and content:
                best = nl + 1 + postal.end()
            break
        words = re.findall(r"[^\W\d_]+", content)
        if any(w.casefold() in _COMMON_LOWER - {"nan"} for w in words):
            break
        pos = line_end
        if re.search(r"\d{3,}", content) or any(w.isupper() and len(w) >= 2 for w in words):
            best = nl + 1 + len(line.rstrip(" \t.,"))
    return best


def _extend_phone_prefix(text: str, spans: List[PIISpan]) -> List[PIISpan]:
    """Include an international prefix ("+1-") written right before a phone span."""
    out = []
    for span in spans:
        if span.kind == "PHONE":
            m = re.search(r"\+\d{1,3}[ \t.-]?$", text[max(0, span.start - 6):span.start])
            if m:
                start = span.start - (len(m.group(0)))
                if all(start >= s.end or s.start >= span.start for s in spans if s is not span):
                    span = PIISpan(start, span.end, span.kind, span.score, span.source)
        out.append(span)
    return out


def _contact_block_orgs(text: str, spans: List[PIISpan]) -> List[PIISpan]:
    """In a contact/signature block, the short line between a person/title line and
    an address line is the employer: "Thomas S.\\nManager\\nHopstop\\n12 Main St"."""
    lines = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        lines.append((offset, offset + len(raw.rstrip("\r\n")), raw.rstrip("\r\n")))
        offset += len(raw)
    covered = lambda a, b, kinds: any(s.start < b and s.end > a and s.kind in kinds for s in spans)
    extra = []
    nonblank = [ln for ln in lines if ln[2].strip(" \t>?")]
    for i in range(1, len(nonblank) - 1):
        start, end, content = nonblank[i]
        body = content.strip(" \t>?")
        lead = len(content) - len(content.lstrip(" \t>?"))
        if (len(body.split()) > 5 or re.search(r"[\d:@]", body)
                or covered(start, end, _PRIORITY.keys())):
            continue
        p_start, p_end, _ = nonblank[i - 1]
        n_start, n_end, n_content = nonblank[i + 1]
        n_lead = len(n_content) - len(n_content.lstrip(" \t>?"))
        if (covered(p_start, p_end, {"PERSON", "TITLE"})
                and any(s.kind == "ADDRESS" and s.start <= n_start + n_lead < s.end + 1
                        and s.start >= n_start for s in spans)):
            extra.append(PIISpan(start + lead, end, "ORGANIZATION", 0.85, "contact_block"))
    return extra


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
            found = _transformer_entities(text, ner_model, ner_threshold, ner_labels, device)
            cased = _truecase(text)
            if cased != text:
                # Second pass on the truecased copy; keep only new, confident spans.
                for span in _transformer_entities(cased, ner_model, max(ner_threshold, 0.85),
                                                  ner_labels, device):
                    if all(span.end <= f.start or span.start >= f.end for f in found):
                        found.append(PIISpan(span.start, span.end, span.kind, span.score, "ner_truecase"))
            entity_spans += found
        except Exception:
            entity_spans += _nltk_entities(text)
    elif ner_backend == "nltk":
        entity_spans += _nltk_entities(text)

    # Entities never override structured values (an email's local part is not a PERSON).
    masks = masked_ranges(text)

    def exposed(candidates):
        if not masks:
            return candidates
        result = []
        for candidate in candidates:
            start = candidate.start
            for left, right in masks:
                if right <= start:
                    continue
                if left >= candidate.end:
                    break
                if start < left and text[start:left].strip():
                    result.append(PIISpan(start, left, candidate.kind, candidate.score, candidate.source))
                start = max(start, right)
            if start < candidate.end and text[start:candidate.end].strip():
                result.append(PIISpan(start, candidate.end, candidate.kind, candidate.score, candidate.source))
        return result

    structured = _resolve_overlaps(exposed(spans))
    cleaned = []
    for span in exposed(entity_spans):
        span = _clean_entity(text, span)
        if span and all(span.end <= s.start or span.start >= s.end for s in structured):
            cleaned.append(span)

    if propagate:
        cleaned += [
            s for s in exposed(_propagate(text, _resolve_overlaps(cleaned)))
            if all(s.end <= t.start or s.start >= t.end for t in structured)
        ]

    final = _consolidate_addresses(text, _resolve_overlaps(structured + cleaned))
    final = _extend_phone_prefix(text, final)
    final += [s for s in exposed(_contact_block_orgs(text, final))
              if all(s.end <= t.start or s.start >= t.end for t in final)]
    return _resolve_overlaps(final)
