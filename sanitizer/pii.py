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
    "ORGANIZATION": 40, "TITLE": 60,
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
    r"|Katu|Tie|Tér|Rue|Strasse|Straße|Weg|Calle|Avenida|Via|Rua|Ulica)"
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
    rf"(?i)\b(?:{_STREET_WORD}(?:straat|strasse|stra\u00dfe|gasse|vej|veien|stien|gata|braut|kuja|katu)"
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
    r"(?i:\b(?:originally\s+from|i\s+live\s+in|i\s+am\s+from|i'm\s+from|live\s+in|lives\s+in|living\s+in|moved\s+to|based\s+in|located\s+in|resident\s+of|born\s+in|hometown\s+is))\s+"
    r"(?P<name>[\wÀ-ÖØ-öø-ÿ'’-]+(?:[ \t]+[\wÀ-ÖØ-öø-ÿ'’-]+){0,2})"
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
        start, end = int(ent["start"]), int(ent["end"])
        if _LABEL_TO_KIND[label] == "PERSON":
            # The token classifier may omit a middle initial or surname.
            # Extend only across adjacent name-shaped tokens.
            suffix = re.match(
                r"(?:\s+(?:(?:[A-ZÀ-ÖØ-Þ]\.)\s+)?"
                r"(?:(?:de|del|der|la|ten|van|von)\s+)?"
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
    return spans


# Titles and workplace affiliations can identify someone in a profile even
# when neither the occupation nor the employer is sensitive in ordinary prose.
_HONORIFIC = re.compile(
    r"\b(?i:mr|mrs|ms|mx|dr|prof)\.(?=[ \t]+[^\W\d_]{2,}\b)"
)
_ROLE_END = (
    r"(?:administrator|analyst|assistant|clerk|consultant|controller|coordinator|"
    r"designer|developer|director|driver|engineer|librarian|manager|mechanic|nurse|"
    r"officer|operator|scientist|specialist|supervisor|teacher|technician|writer)"
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

    return _resolve_overlaps(structured + cleaned)
