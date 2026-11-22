"""Consistent, type-preserving surrogate values for detected sensitive spans.

Three properties matter for utility and privacy:

* **Consistency** – the same original always maps to the same surrogate within
  a session, so "John ... John's" stays coherent ("Alex ... Alex's") and a
  multi-turn conversation keeps referring to the same (fake) person.
* **Format preservation** – phone numbers, IDs and keys keep their shape, and
  dates are shifted by one session-wide offset so intervals between dates
  (ages, durations) survive sanitization.
* **Determinism** – values derive from a keyed SHA-256, never Python's salted
  ``hash()``, so the same input gives the same output across processes.
"""

from __future__ import annotations

import hashlib
import random
import re
import string
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple

_FIRST_NAMES = [
    "Alex", "Jordan", "Sam", "Morgan", "Casey", "Riley", "Taylor", "Jamie",
    "Avery", "Quinn", "Drew", "Skyler", "Rowan", "Parker", "Reese", "Emerson",
    "Hayden", "Kendall", "Logan", "Peyton", "Sawyer", "Blake", "Cameron", "Dana",
]
_LAST_NAMES = [
    "Morgan", "Bennett", "Carter", "Ellis", "Foster", "Hayes", "Keller", "Lane",
    "Mercer", "Nolan", "Porter", "Reed", "Sutton", "Walsh", "Hale", "Brooks",
    "Chandler", "Dalton", "Harper", "Monroe",
]
_CITIES = [
    "Springfield", "Riverside", "Fairview", "Greenville", "Madison", "Franklin",
    "Clinton", "Georgetown", "Ashford", "Brookfield", "Lakewood", "Milton",
]
_ORGS = [
    "Acme Corp", "Globex", "Initech", "Northwind", "Contoso", "Umbrella Labs",
    "Vandelay Industries", "Hooli", "Stark Solutions", "Wayne Group",
]
_STREETS = ["Main", "Elm", "Park", "Maple", "Cedar", "Pine", "Oak", "Lake", "Hill", "Willow"]

_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
_MONTH_LOOKUP = {m.lower(): i + 1 for i, m in enumerate(_MONTHS)}
_MONTH_LOOKUP.update({m[:3].lower(): i + 1 for i, m in enumerate(_MONTHS)})
_MONTH_LOOKUP["sept"] = 9


def _stable_rng(*parts: str) -> random.Random:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _ordinal(day: int) -> str:
    if 11 <= day % 100 <= 13:
        return f"{day}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def match_case(template: str, word: str) -> str:
    """Give ``word`` the capitalisation pattern of ``template``."""
    if template.isupper() and len(template) > 1:
        return word.upper()
    if template[:1].isupper():
        return word[:1].upper() + word[1:]
    if template.islower():
        return word.lower()
    return word


def _replace_chars(value: str, rng: random.Random, keep_prefix: int = 0) -> str:
    """Replace digits with digits and letters with letters, keeping separators."""
    out = []
    for i, ch in enumerate(value):
        if i < keep_prefix:
            out.append(ch)
        elif ch.isdigit():
            out.append(str(rng.randint(0, 9)))
        elif ch.isalpha() and ch.isascii():
            pool = string.ascii_uppercase if ch.isupper() else string.ascii_lowercase
            out.append(rng.choice(pool))
        else:
            out.append(ch)
    return "".join(out)


# Structural words a surrogate may legitimately share with the original
# ("12 Oak Street" -> "40 Elm Street"); they carry no identifying information.
_NON_IDENTIFYING = {
    "street", "st", "avenue", "ave", "boulevard", "blvd", "road", "rd", "drive", "dr",
    "lane", "ln", "way", "place", "pl", "court", "ct", "circle", "cir", "terrace", "ter",
    "trail", "trl", "parkway", "pkwy", "highway", "hwy", "str", "suite", "apt", "apartment",
    "unit", "flat", "house", "plot", "box", "the", "and", "com", "www", "http", "https",
    "example", "inc", "llc", "ltd", "group",
}


def leaks_fragment(original: str, surrogate: str) -> bool:
    """True if ``surrogate`` contains an identifying token of ``original``.

    A token counts if it has a digit or more than two letters, and is not
    a structural word such as "Street". Comparison is whole-token and
    case-insensitive, so "192" leaks into "192.0.2.5" but "12" does not
    leak into "120".
    """
    tokens = {
        token.casefold() for token in re.findall(r"\w+", original)
        if (len(token) > 2 or any(ch.isdigit() for ch in token))
        and token.casefold() not in _NON_IDENTIFYING
    }
    surrogate_tokens = {token.casefold() for token in re.findall(r"\w+", surrogate)}
    return bool(tokens & surrogate_tokens)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


class SurrogateGenerator:
    """Maps (kind, original) pairs to surrogates, remembering every decision."""

    def __init__(self, style: str = "realistic", secret: str = "prosan"):
        if style not in ("realistic", "placeholder"):
            raise ValueError("surrogate style must be 'realistic' or 'placeholder'")
        self.style = style
        self.secret = secret
        self._forward: Dict[Tuple[str, str], str] = {}
        self._reverse: Dict[str, str] = {}
        self._counters: Dict[str, int] = {}
        # One date shift per session keeps the distance between dates intact.
        self._date_shift = timedelta(days=_stable_rng(secret, "date-shift").choice(
            [d for d in range(-240, 241) if abs(d) >= 45]
        ))

    # ── public API ──────────────────────────────────────────────
    @property
    def mapping(self) -> Dict[str, str]:
        """surrogate -> original"""
        return dict(self._reverse)

    def reserve_placeholders(self, values):
        """Avoid mapping an existing numbered marker to a newly detected value."""
        for value in values:
            match = re.fullmatch(r"<([A-Z_]+)_(\d{1,9})>", value)
            if match:
                kind, index = match.groups()
                self._counters[kind] = max(self._counters.get(kind, 0), int(index))

    def lookup(self, kind: str, original: str) -> Optional[str]:
        return self._forward.get((kind, self._key(kind, original)))

    def get(self, kind: str, original: str) -> str:
        key = (kind, self._key(kind, original))
        if key in self._forward:
            return self._forward[key]

        if self.style == "placeholder" or kind == "TITLE":
            self._counters[kind] = self._counters.get(kind, 0) + 1
            surrogate = f"<{kind}_{self._counters[kind]}>"
        else:
            surrogate = self._realistic(kind, original)
            # Kinds built from a seeded character shuffle can coincidentally
            # reproduce a group of the original ("71-33-52-22" -> "22-..."),
            # which re-exposes part of it. Reseed until no fragment survives.
            base_secret = self.secret
            try:
                for attempt in range(1, 40):
                    if not leaks_fragment(original, surrogate):
                        break
                    self.secret = f"{base_secret}~{attempt}"
                    surrogate = self._realistic(kind, original)
            finally:
                self.secret = base_secret

        self.register(kind, original, surrogate)
        if kind == "PERSON" and self.style == "realistic":
            self._register_name_parts(original, surrogate)
        return surrogate

    def register(self, kind: str, original: str, surrogate: str) -> None:
        self._forward[(kind, self._key(kind, original))] = surrogate
        self._reverse.setdefault(surrogate, original)

    # ── internals ───────────────────────────────────────────────
    @staticmethod
    def _key(kind: str, original: str) -> str:
        if kind in ("PERSON", "LOCATION", "ORGANIZATION", "EMAIL", "CONTEXTUAL"):
            return original.strip().lower()
        return original.strip()

    def _pick(self, pool, kind: str, original: str, attempt: int = 0):
        return _stable_rng(self.secret, kind, original.lower(), str(attempt)).choice(pool)

    def _unused(self, make, original: str, limit: int = 40) -> str:
        """Call make(attempt) until it yields a value that is not used for another
        original and does not reuse any identifying fragment of this one."""
        value = make(0)
        for attempt in range(1, limit):
            owner = self._reverse.get(value)
            if ((owner is None or owner.lower() == original.lower())
                    and value.lower() != original.lower()
                    and not leaks_fragment(original, value)):
                return value
            value = make(attempt)
        return value

    def _register_name_parts(self, original: str, surrogate: str) -> None:
        """"John Doe" -> "Alex Reed" also means a later bare "John" -> "Alex"."""
        o_parts, s_parts = original.split(), surrogate.split()
        if len(o_parts) == len(s_parts) and len(o_parts) > 1:
            for o, s in zip(o_parts, s_parts):
                key = ("PERSON", o.lower())
                if len(o) > 1 and key not in self._forward:
                    self.register("PERSON", o, s)

    def _realistic(self, kind: str, original: str) -> str:
        rng = _stable_rng(self.secret, kind, original)

        if kind == "PERSON":
            n_parts = len(original.split())
            source_words = {word.casefold() for word in re.findall(r"\w+", original)}
            first_names = [name for name in _FIRST_NAMES if name.casefold() not in source_words]
            last_names = [name for name in _LAST_NAMES if name.casefold() not in source_words]

            def make(attempt):
                first = self._pick(first_names or _FIRST_NAMES, "first", original, attempt)
                if n_parts == 1:
                    return first
                last = self._pick(last_names or _LAST_NAMES, "last", original, attempt)
                return f"{first} {last}"
            return self._unused(make, original)

        if kind == "LOCATION":
            source_words = {word.casefold() for word in re.findall(r"\w+", original)}
            cities = [city for city in _CITIES
                      if not source_words.intersection(word.casefold() for word in re.findall(r"\w+", city))]
            return self._unused(lambda a: self._pick(cities or _CITIES, kind, original, a), original)

        if kind == "NRP":
            pool = ["Canadian", "Kenyan", "Chilean", "Norwegian", "Malaysian", "Portuguese",
                    "Ghanaian", "Estonian", "Peruvian", "Icelandic"]
            plural = original.endswith("s") and not original.endswith("ss")
            source = {original.casefold().rstrip("s")}

            def make(attempt):
                choice = self._pick([p for p in pool if p.casefold() not in source] or pool,
                                    kind, original, attempt)
                if plural and not choice.endswith(("ese", "ss")):
                    choice += "s"
                return match_case(original, choice)
            return self._unused(make, original)

        if kind == "ORGANIZATION":
            source_words = {word.casefold() for word in re.findall(r"\w+", original)}
            organizations = [org for org in _ORGS
                             if not source_words.intersection(word.casefold() for word in re.findall(r"\w+", org))]
            if not organizations:
                organizations = [f"{first} {last} Group"
                                 for first in _FIRST_NAMES for last in _LAST_NAMES
                                 if first.casefold() not in source_words
                                 and last.casefold() not in source_words
                                 and "group" not in source_words]
            return self._unused(lambda a: self._pick(organizations or _ORGS, kind, original, a), original)

        if kind == "EMAIL":
            def make(attempt):
                first = self._pick(_FIRST_NAMES, "first", original, attempt).lower()
                last = self._pick(_LAST_NAMES, "last", original, attempt).lower()
                return f"{first}.{last}@example.com"
            return self._unused(make, original)

        if kind == "URL":
            scheme = "https://" if original.lower().startswith("https") else "http://"
            path = re.sub(r"^[a-z]+://|^www\.", "", original, flags=re.I).partition("/")[2]
            path = path.split("?")[0].split("#")[0]
            # Keep purely descriptive path segments ("admin", "docs"); drop the rest.
            keep = [seg for seg in path.split("/") if seg.isalpha() and len(seg) <= 20]
            suffix = "/" + "/".join(keep) if keep else ""
            return self._unused(
                lambda a: f"{scheme}example{'' if a == 0 else a + 1}.com{suffix}", original
            )

        if kind == "IP_ADDRESS":
            if ":" in original:
                return self._unused(lambda a: f"2001:db8::{_stable_rng(original, str(a)).randint(1, 0xFFFF):x}", original)
            # RFC 5737 documentation ranges: never routable, clearly not real.
            # Rotating across all three lets the leak guard avoid an original
            # that itself contains 192, 198, 203, ... as one of its groups.
            ranges = ("192.0.2", "198.51.100", "203.0.113")
            return self._unused(
                lambda a: f"{ranges[a % 3]}.{_stable_rng(self.secret, original, str(a)).randint(1, 254)}",
                original,
            )

        if kind == "MAC_ADDRESS":
            return _replace_chars(original, rng)

        if kind == "CARD_NUMBER":
            # Same grouping; digits chosen so the result fails the Luhn check
            # and therefore can never be a real card number.
            value = _replace_chars(original, rng)
            digits = re.sub(r"\D", "", value)
            if _luhn_ok(digits):
                last = value.rstrip()[-1]
                bumped = str((int(last) + 1) % 10)
                value = value[: value.rfind(last)] + bumped + value[value.rfind(last) + 1:]
            return value

        if kind == "PHONE":
            # Change every numeric group. Keeping a country code or an unchanged
            # short group leaves part of the labeled phone identifier exposed.
            def make(attempt):
                candidate = _replace_chars(
                    original, _stable_rng(self.secret, kind, original, str(attempt))
                )
                old_groups = re.findall(r"\d+", original)
                new_groups = re.findall(r"\d+", candidate)
                if any(old == new for old, new in zip(old_groups, new_groups)):
                    # Deterministically alter one digit in each retained group.
                    chars = list(candidate)
                    for old, new, match in zip(old_groups, new_groups, re.finditer(r"\d+", candidate)):
                        if old == new:
                            index = match.end() - 1
                            chars[index] = str((int(chars[index]) + 1) % 10)
                    candidate = "".join(chars)
                return candidate

            return self._unused(make, original)

        if kind == "DATE":
            shifted = self._shift_date(original)
            if shifted:
                return shifted
            return _replace_chars(original, rng)

        if kind == "DATE_TIME":
            match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})([ T])(\d{2}:\d{2})(:\d{2})?", original.strip())
            if match:
                shifted_date = self._shift_date(match.group(1)) or match.group(1)
                time_format = "%H:%M:%S" if match.group(4) else "%H:%M"
                time_value = datetime.strptime(match.group(3) + (match.group(4) or ""), time_format)
                minute_shift = rng.choice([n for n in range(-180, 181) if n])
                shifted_time = (time_value + timedelta(minutes=minute_shift)).strftime(time_format)
                return f"{shifted_date}{match.group(2)}{shifted_time}"
            return _replace_chars(original, rng)

        if kind == "AGE":
            try:
                age = int(original)
            except ValueError:
                return _replace_chars(original, rng)
            width = len(original)
            upper = min(120, (10 ** width) - 1)
            choices = [candidate for candidate in range(1, upper + 1) if candidate != age]
            if not choices:
                return _replace_chars(original, rng)
            value = rng.choice(choices)
            return f"{value:0{width}d}"

        if kind == "ADDRESS":
            source_words = {word.casefold() for word in re.findall(r"\w+", original)}
            suffixes = {"street", "st", "avenue", "ave", "boulevard", "blvd", "road", "rd",
                        "drive", "dr", "lane", "ln", "way", "place", "court", "circle",
                        "terrace", "trail", "parkway", "highway"}
            last_word = original.split()[-1].rstrip(".,") if original.split() else ""
            suffix = last_word if last_word.casefold() in suffixes else "Street"

            def make(attempt):
                number = self._pick(tuple(str(n) for n in range(10, 1000)), "address-number", original, attempt)
                available = [name for name in _STREETS if name.casefold() not in source_words]
                if not available:
                    available = list(_STREETS)
                street = self._pick(available, "address-street", original, attempt)
                return f"{number} {street} {suffix}"

            return self._unused(make, original)

        if kind in ("API_KEY", "SECRET", "PASSWORD", "PRIVATE_KEY", "JWT"):
            if kind == "PRIVATE_KEY":
                return "-----BEGIN PRIVATE KEY-----\n[REDACTED]\n-----END PRIVATE KEY-----"
            # Keep a recognisable vendor prefix (sk-, AKIA, ghp_) for utility.
            m = re.match(r"^(?:sk|pk|rk)[-_](?:live|test|proj)?[-_]?|^AKIA|^gh[pousr]_|^xox[baprs]-|^AIza|^hf_|^eyJ", original)
            return _replace_chars(original, rng, keep_prefix=m.end() if m else 0)

        if kind == "USERNAME":
            return self._unused(lambda a: f"user{_stable_rng(original, str(a)).randint(100, 999)}", original)

        if kind == "HOSTNAME":
            return self._unused(lambda a: f"db{_stable_rng(original, str(a)).randint(1, 99)}.example.internal", original)

        # SSN, AADHAAR, PAN, IBAN, ID_NUMBER, POSTAL_CODE, ...: same shape, new characters.
        keep = 2 if kind == "IBAN" else 0
        return _replace_chars(original, rng, keep_prefix=keep)

    def _shift_date(self, original: str) -> Optional[str]:
        """Parse the date formats the detector emits and shift by the session offset."""
        text = original.strip()

        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if text.casefold() in {day.casefold() for day in weekdays}:
            weekday = next(i for i, day in enumerate(weekdays) if day.casefold() == text.casefold())
            shift = abs(self._date_shift.days) % 6 + 1
            return match_case(text, weekdays[(weekday + shift) % 7])

        # 2024-01-15
        m = re.fullmatch(r"(\d{4})([-/.])(\d{1,2})\2(\d{1,2})", text)
        if m:
            d = self._safe_date(int(m[1]), int(m[3]), int(m[4]))
            if d:
                d += self._date_shift
                return f"{d.year}{m[2]}{d.month:0{len(m[3])}d}{m[2]}{d.day:0{len(m[4])}d}"

        # 05/12/2024 (month-first when valid, else day-first)
        m = re.fullmatch(r"(\d{1,2})([-/.])(\d{1,2})\2(\d{2,4})", text)
        if m:
            year = int(m[4]) + (2000 if len(m[4]) == 2 else 0)
            a, b = int(m[1]), int(m[3])
            month_first = a <= 12
            d = self._safe_date(year, a, b) if month_first else self._safe_date(year, b, a)
            if d:
                d += self._date_shift
                first, second = (d.month, d.day) if month_first else (d.day, d.month)
                y = str(d.year)[-len(m[4]):]
                return f"{first:0{len(m[1])}d}{m[2]}{second:0{len(m[3])}d}{m[2]}{y}"

        # July 2nd, 1995 / July 2 / Jul 2nd
        m = re.fullmatch(r"([A-Za-z]+)\.?\s+(\d{1,2})(st|nd|rd|th)?(,?\s*)(\d{4})?", text)
        if m and m[1].lower() in _MONTH_LOOKUP:
            year = int(m[5]) if m[5] else 2001
            d = self._safe_date(year, _MONTH_LOOKUP[m[1].lower()], int(m[2]))
            if d:
                d += self._date_shift
                month = _MONTHS[d.month - 1] if len(m[1]) > 3 else _MONTHS[d.month - 1][:3]
                day = _ordinal(d.day) if m[3] else str(d.day)
                tail = f"{m[4]}{d.year}" if m[5] else ""
                return f"{match_case(m[1], month)} {day}{tail}"

        # 2nd July / 15 March 1990
        m = re.fullmatch(r"(\d{1,2})(st|nd|rd|th)?\s+(of\s+)?([A-Za-z]+)(,?\s*)(\d{4})?", text)
        if m and m[4].lower() in _MONTH_LOOKUP:
            year = int(m[6]) if m[6] else 2001
            d = self._safe_date(year, _MONTH_LOOKUP[m[4].lower()], int(m[1]))
            if d:
                d += self._date_shift
                month = _MONTHS[d.month - 1] if len(m[4]) > 3 else _MONTHS[d.month - 1][:3]
                day = _ordinal(d.day) if m[2] else str(d.day)
                tail = f"{m[5]}{d.year}" if m[6] else ""
                return f"{day} {m[3] or ''}{match_case(m[4], month)}{tail}"

        return None

    @staticmethod
    def _safe_date(year: int, month: int, day: int) -> Optional[date]:
        try:
            return date(year, month, day)
        except ValueError:
            return None
