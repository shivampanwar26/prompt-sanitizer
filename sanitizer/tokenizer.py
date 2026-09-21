import re
from typing import List, Dict

# A word is a run of word characters, optionally joined by inner punctuation so
# that "home-care", "don't", "john@example.com" and "os.path" stay single units.
_WORD_RE = re.compile(r"\w+(?:['’.@+\-]\w+)*")


def extract_words(text: str) -> List[Dict]:
    """
    Simple word spans.

    This is deliberately independent of the model tokenizer. The model modules
    later map these character spans to model tokens using offset_mapping.
    """
    return [
        {"word": match.group(0), "start": match.start(), "end": match.end()}
        for match in _WORD_RE.finditer(text)
    ]
