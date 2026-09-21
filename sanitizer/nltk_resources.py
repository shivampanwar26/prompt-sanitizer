"""Lazy, fail-soft access to NLTK data (downloaded once, on first use)."""

from functools import lru_cache

try:
    import nltk
except ImportError:  # pragma: no cover - nltk is a hard requirement, but degrade gracefully
    nltk = None


@lru_cache(maxsize=None)
def _available(resource: str) -> bool:
    if nltk is None:
        return False
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        try:
            nltk.download(resource.split("/")[-1], quiet=True)
            nltk.data.find(resource)
            return True
        except Exception:
            return False


def ensure(*resources: str) -> bool:
    """Return True when every resource (e.g. "corpora/wordnet") is usable."""
    return all(_available(r) for r in resources)


@lru_cache(maxsize=1)
def _wordnet():
    """NLTK ships WordNet zipped and its reader loads straight from the zip,
    so ``nltk.data.find("corpora/wordnet")`` never succeeds even when the
    corpus works fine — probe it directly instead."""
    if nltk is None:
        return None
    from nltk.corpus import wordnet
    try:
        wordnet.synsets("test")
        return wordnet
    except LookupError:
        try:
            nltk.download("wordnet", quiet=True)
            wordnet.synsets("test")
            return wordnet
        except Exception:
            return None


@lru_cache(maxsize=50000)
def is_dictionary_word(word: str) -> bool:
    """True if the lower-cased word, or every part of a hyphenated compound
    ("home-care", "well-being"), is an English dictionary word."""
    lower = word.lower()
    wn = _wordnet()
    if wn is None:
        return word.isalpha() and word.islower()
    if wn.synsets(lower):
        return True
    if "-" in lower:
        parts = [p for p in lower.split("-") if p]
        return bool(parts) and all(wn.synsets(p) for p in parts)
    return False


def pos_tag(tokens):
    if nltk is None or not ensure("taggers/averaged_perceptron_tagger_eng"):
        return ["NN"] * len(tokens)
    try:
        return [tag for _, tag in nltk.pos_tag(tokens)]
    except Exception:
        return ["NN"] * len(tokens)
