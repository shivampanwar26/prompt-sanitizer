import torch
from .model_registry import get_causal_model_and_tokenizer


class PrivacyCalculator:
    """
    ProSan Contextual Privacy Calculator (Section IV-B & VII-A).

    Calculates self-information:
        I(t_i) = -log2 P(t_i | previous tokens)

    Includes repeated entity max-pooling (Section VII-A).
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
    def token_information(self, text: str):
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        input_ids = encoded["input_ids"].to(self.device)

        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]

        log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_log_probs = log_probs.gather(
            2, targets.unsqueeze(-1)
        ).squeeze(-1)

        info = -token_log_probs / torch.log(torch.tensor(2.0, device=self.device))

        info = torch.cat(
            [torch.zeros((1, 1), device=self.device), info],
            dim=1,
        )

        return (
            info.squeeze(0).cpu().tolist(),
            encoded["offset_mapping"].squeeze(0).cpu().tolist(),
        )

    def calculate(self, text: str, words):
        if not words:
            return [], []

        token_info, offsets = self.token_information(text)

        raw_scores = []
        for word in words:
            total = 0.0
            for i, (start, end) in enumerate(offsets):
                if end <= word["start"] or start >= word["end"]:
                    continue
                total += token_info[i]

            raw_scores.append(total)

        word_max_privacy = {}
        for i, word in enumerate(words):
            w_lower = word["word"].lower()
            if w_lower not in word_max_privacy or raw_scores[i] > word_max_privacy[w_lower]:
                word_max_privacy[w_lower] = raw_scores[i]

        pooled_raw_scores = [
            word_max_privacy[w["word"].lower()] for w in words
        ]

        normalized = normalize(pooled_raw_scores)
        return normalized, pooled_raw_scores


def normalize(values):
    if not values:
        return []

    lo = min(values)
    hi = max(values)

    if hi == lo:
        return [0.0 for _ in values]

    return [(x - lo) / (hi - lo) for x in values]
