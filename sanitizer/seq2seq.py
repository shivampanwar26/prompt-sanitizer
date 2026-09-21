"""Local seq2seq training and inference for low-resource sanitization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class _PairDataset(Dataset):
    def __init__(self, examples: List[Dict[str, str]], tokenizer, max_source_length: int, max_target_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example["original"], max_length=self.max_source_length,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=example["sanitized"], max_length=self.max_target_length,
            truncation=True,
        )["input_ids"]
        encoded["labels"] = labels
        return encoded


def _load_examples(dataset_path: str | Path) -> List[Dict[str, str]]:
    with Path(dataset_path).open("r", encoding="utf-8") as handle:
        examples = [json.loads(line) for line in handle if line.strip()]
    if not examples:
        raise ValueError("Training dataset is empty.")
    if any(not isinstance(item.get("original"), str) or not isinstance(item.get("sanitized"), str) for item in examples):
        raise ValueError("Every JSONL row must contain string 'original' and 'sanitized' fields.")
    return examples


def train_seq2seq(
    dataset_path: str | Path,
    output_dir: str | Path,
    base_model: str = "google/flan-t5-small",
    epochs: float = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_source_length: int = 256,
    max_target_length: int = 256,
) -> Path:
    """Fine-tune and save a local sanitizer model from generated JSONL pairs."""
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")

    # Kept local so importing the dataset generator never loads Trainer or its
    # optional integrations.
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    examples = _load_examples(dataset_path)
    destination = Path(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    dataset = _PairDataset(examples, tokenizer, max_source_length, max_target_length)

    args = TrainingArguments(
        output_dir=str(destination),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=max(1, min(10, len(dataset))),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()
    destination.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(str(destination))
    return destination


class LocalSeq2SeqSanitizer:
    """Run a fine-tuned sanitizer entirely from a local model directory."""

    def __init__(self, model_path: str | Path, device: str = "auto"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def sanitize(self, prompt: str, max_new_tokens: int = 256) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        output_ids = self.model.generate(**encoded, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
