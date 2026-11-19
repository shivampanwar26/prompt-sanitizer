import json

from sanitizer.config import SanitizerConfig
from sanitizer.dataset import generate_dataset_from_file


def test_dataset_generation_writes_source_target_pairs(tmp_path):
    source = tmp_path / "prompts.txt"
    output = tmp_path / "training.jsonl"
    source.write_text("Email me@example.com.\nI have a fever.\n", encoding="utf-8")

    count = generate_dataset_from_file(source, output, SanitizerConfig(mode="pii_only"))

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert count == 2
    assert rows[0]["original"] == "Email me@example.com."
    assert rows[0]["sanitized"] == "Email <EMAIL>."
    assert rows[1]["sanitized"] == "I have a fever."
