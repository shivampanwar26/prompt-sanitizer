import math
from typing import List, Dict, Optional
import torch
from .model_registry import get_causal_model_and_tokenizer


class PromptEvaluator:
    """
    Evaluator for ProSan Prompt Sanitization (Section VI) — Enhanced.

    1. Readability / Perplexity evaluation with batched support.
    2. Privacy Hiding Rate (PHR).
    """

    def __init__(self, model_name="distilgpt2", device="auto", model_obj=None, tokenizer_obj=None):
        if model_obj is not None and tokenizer_obj is not None:
            self.model = model_obj
            self.tokenizer = tokenizer_obj
            self.device = next(self.model.parameters()).device
        else:
            self.model, self.tokenizer = get_causal_model_and_tokenizer(model_name, device)
            self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0

        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = encoded["input_ids"]

        if input_ids.shape[1] < 2:
            return 1.0

        outputs = self.model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss.item()
        ppl = math.exp(neg_log_likelihood)

        return round(ppl, 4)

    @torch.no_grad()
    def perplexity_batch(self, texts: List[str]) -> List[float]:
        """Compute perplexity for multiple texts in a single batched forward pass."""
        if not texts:
            return []

        # Filter out empty texts
        valid = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not valid:
            return [0.0] * len(texts)

        # Tokenize with padding
        valid_texts = [t for _, t in valid]
        encoded = self.tokenizer(
            valid_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        results = [0.0] * len(texts)

        for batch_idx, (orig_idx, _) in enumerate(valid):
            # Compute per-sequence loss (ignore padding)
            seq_mask = attention_mask[batch_idx]
            seq_len = seq_mask.sum().item()

            if seq_len < 2:
                results[orig_idx] = 1.0
                continue

            seq_logits = logits[batch_idx, :seq_len - 1, :]
            seq_labels = input_ids[batch_idx, 1:seq_len]

            loss = torch.nn.functional.cross_entropy(
                seq_logits, seq_labels, reduction="mean"
            )
            results[orig_idx] = round(math.exp(loss.item()), 4)

        return results

    @staticmethod
    def calculate_phr(original_sensitive_items: List[str], sanitized_text: str) -> Dict[str, float]:
        if not original_sensitive_items:
            return {"total": 0, "hidden": 0, "phr": 100.0}

        total = len(original_sensitive_items)
        hidden = 0

        for item in original_sensitive_items:
            if item.lower() not in sanitized_text.lower():
                hidden += 1

        phr = (hidden / total) * 100.0 if total > 0 else 100.0
        return {
            "total": total,
            "hidden": hidden,
            "phr": round(phr, 2),
        }
