"""Calibrated privacy risk O_w (Section IV-B, VII-A) — Enhanced.

The paper min-max normalises self-information inside each prompt, which
guarantees that *some* word in every prompt scores 1.0: harmless prompts such
as "I have a fever" still get a word rewritten, and rare-but-public terms
("dyspnea") look as private as a name.

Here risk is absolute and combines two independent signals:

    O_w = prior(w) * (1 - exp(-I_w / kappa))

* ``1 - exp(-I/kappa)`` maps surprisal in bits onto [0, 1) without looking at
  the other words in the prompt, so the threshold means the same thing for
  every input.
* ``prior(w)`` is a *specificity* prior from the word's form: a capitalised
  non-initial token that is not a dictionary word ("Estrella") is very likely
  identifying; a lower-case dictionary word ("fever", "dyspnea") is not, no
  matter how surprising.

Repeated mentions are max-pooled (Section VII-A): a name's later mentions are
predictable from the first, but they are exactly as private.
"""

import math
import re
from typing import Dict, List, Sequence, Tuple

from .nltk_resources import is_dictionary_word

def is_sentence_initial(text: str, start: int) -> bool:
    """True when only whitespace/quotes/brackets separate ``start`` from a sentence boundary."""
    before = text[:start].rstrip(" \t\"'“‘([")
    return not before or before[-1] in ".!?:\n"


_VERSIONED_NAME_RE = re.compile(r"^[A-Za-z]{3,}[0-9]{1,2}$")  # boto3, gpt4, oauth2, web3, python3


def privacy_prior(word: str, pos_tag: str, sentence_initial: bool) -> float:
    digits = sum(ch.isdigit() for ch in word)
    if digits:
        if _VERSIONED_NAME_RE.match(word):
            return 0.2   # a library/format name with a trailing version digit, not an identifier
        # Mixed alphanumerics and long numbers identify; "2" or "10" rarely do.
        return 0.9 if (digits >= 4 or digits < len(word)) else 0.4
    if word.isupper() and len(word) > 1:
        return 0.35          # acronyms: ICU, MRI, API
    if word[:1].isupper():
        in_dict = is_dictionary_word(word)
        if sentence_initial:
            # Capitalisation here is English orthography, not identity evidence.
            # POS taggers routinely mistag a sentence-initial verb/noun as NNP
            # ("Translate this" -> NNP); a genuine dictionary word should score
            # the same regardless of that mistag, so only an unrecognised token
            # ("Estrella opened...") is treated as a likely name.
            return 0.15 if in_dict else 0.85
        return 0.8 if in_dict else 1.0
    if is_dictionary_word(word):
        if pos_tag.startswith("NN"):
            return 0.3
        if pos_tag.startswith("CD"):
            return 0.4
        return 0.2
    return 0.7               # unknown lower-case token: handle, username, rare surname


def surprisal_risk(bits: float, saturation: float) -> float:
    return 1.0 - math.exp(-max(bits, 0.0) / max(saturation, 1e-6))


class PrivacyCalculator:
    def __init__(self, saturation: float = 8.0):
        self.saturation = saturation

    def calculate(self, text: str, words: List[Dict], bits: Sequence[float],
                  pos_tags: Sequence[str]) -> Tuple[List[float], List[float]]:
        """Return (O_w in [0, 1], pooled self-information bits) for each word."""
        pooled_bits: Dict[str, float] = {}
        for word, b in zip(words, bits):
            key = word["word"].lower()
            pooled_bits[key] = max(pooled_bits.get(key, 0.0), b)

        risk, raw = [], []
        pooled_risk: Dict[str, float] = {}
        for word, tag in zip(words, pos_tags):
            key = word["word"].lower()
            b = pooled_bits[key]
            prior = privacy_prior(word["word"], tag, is_sentence_initial(text, word["start"]))
            o = prior * surprisal_risk(b, self.saturation)
            pooled_risk[key] = max(pooled_risk.get(key, 0.0), o)
            raw.append(b)
            risk.append(o)
        # A word is as private as its most identifying mention.
        return [pooled_risk[w["word"].lower()] for w in words], raw
