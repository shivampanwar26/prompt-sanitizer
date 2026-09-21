import math
import re
from typing import Dict, List

import torch

from .model_registry import get_causal_model_and_tokenizer


def _contains(text: str, item: str) -> bool:
    """Case-insensitive whole-token containment ("Sam" is not found in "Samsung")."""
    pattern = (r"(?<!\w)" if item[:1].isalnum() else "") + re.escape(item) + (r"(?!\w)" if item[-1:].isalnum() else "")
    return re.search(pattern, text, re.I) is not None


class PromptEvaluator:
    """
    Evaluator for ProSan Prompt Sanitization (Section VI).

    1. Readability: perplexity under the causal LM (batched).
    2. Privacy Hiding Rate (PHR): share of gold sensitive items absent from the output.
    3. Utility retention: share of task-critical words that survive sanitization.
    """

    def __init__(self, model_name="distilgpt2", device="auto", model_obj=None, tokenizer_obj=None):
        if model_obj is not None and tokenizer_obj is not None:
            self.model, self.tokenizer = model_obj, tokenizer_obj
        else:
            self.model, self.tokenizer = get_causal_model_and_tokenizer(model_name, device)
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        return self.perplexity_batch([text])[0]

    @torch.no_grad()
    def perplexity_batch(self, texts: List[str]) -> List[float]:
        """Compute perplexity for multiple texts in a single batched forward pass."""
        results = [0.0] * len(texts)
        valid = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not valid:
            return results

        encoded = self.tokenizer(
            [t for _, t in valid], return_tensors="pt", padding=True, truncation=True, max_length=512,
        ).to(self.device)
        input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]
        logits = self.model(input_ids, attention_mask=attention_mask).logits

        for batch_idx, (orig_idx, _) in enumerate(valid):
            seq_len = int(attention_mask[batch_idx].sum().item())
            if seq_len < 2:
                results[orig_idx] = 1.0
                continue
            loss = torch.nn.functional.cross_entropy(
                logits[batch_idx, :seq_len - 1].float(), input_ids[batch_idx, 1:seq_len]
            )
            results[orig_idx] = round(math.exp(loss.item()), 4)
        return results

    @staticmethod
    def calculate_phr(original_sensitive_items: List[str], sanitized_text: str) -> Dict[str, float]:
        items = [i for i in dict.fromkeys(original_sensitive_items) if i and i.strip()]
        if not items:
            return {"total": 0, "hidden": 0, "phr": 100.0}
        hidden = sum(not _contains(sanitized_text, item) for item in items)
        return {"total": len(items), "hidden": hidden, "phr": round(100.0 * hidden / len(items), 2)}

    @staticmethod
    def utility_retention(keep_items: List[str], sanitized_text: str) -> Dict[str, float]:
        items = [i for i in dict.fromkeys(keep_items) if i and i.strip()]
        if not items:
            return {"total": 0, "kept": 0, "retention": 100.0}
        kept = sum(_contains(sanitized_text, item) for item in items)
        return {"total": len(items), "kept": kept, "retention": round(100.0 * kept / len(items), 2)}
