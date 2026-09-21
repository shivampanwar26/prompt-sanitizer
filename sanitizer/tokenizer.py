from typing import List, Dict


def extract_words(text: str) -> List[Dict]:
    """
    Simple word spans.

    This is deliberately independent of the model tokenizer. The model modules
    later map these character spans to model tokens using offset_mapping.
    """
    import re

    result = []

    for match in re.finditer(r"\b[\w@.+-]+\b", text):
        result.append(
            {
                "word": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
        )

    return result
