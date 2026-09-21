"""Create training pairs for the diagram's sanitized-dataset stage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .config import SanitizerConfig
from .sanitizer import PromptSanitizer


@dataclass
class SanitizedExample:
    """One source/target pair suitable for sequence-to-sequence training."""

    original: str
    sanitized: str
    selected_words: list


def read_prompt_lines(path: str | Path) -> List[str]:
    """Read one prompt per non-empty line from a UTF-8 text file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def generate_sanitized_dataset(
    prompts: Iterable[str],
    output_path: str | Path,
    config: Optional[SanitizerConfig] = None,
    include_unchanged: bool = True,
) -> int:
    """Write JSONL ``original`` → ``sanitized`` examples and return its size.

    Keeping unchanged examples by default is important: it trains the deployed
    model to preserve ordinary content rather than rewriting every prompt.
    """
    sanitizer = PromptSanitizer(config=config)
    examples = []
    for prompt in prompts:
        result = sanitizer.sanitize(prompt)
        if include_unchanged or result.text != prompt:
            examples.append(SanitizedExample(prompt, result.text, result.selected_words))

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
    return len(examples)


def generate_dataset_from_file(
    input_path: str | Path,
    output_path: str | Path,
    config: Optional[SanitizerConfig] = None,
    include_unchanged: bool = True,
) -> int:
    return generate_sanitized_dataset(
        read_prompt_lines(input_path), output_path, config, include_unchanged
    )
