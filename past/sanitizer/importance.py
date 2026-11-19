import torch
from .model_registry import get_causal_model_and_tokenizer


class ImportanceCalculator:
    """
    ProSan utility importance calculation — Enhanced.

    Default: Attention-based importance (1 forward pass, no autograd).
    Uses last-layer attention weights to measure how much the model
    relies on each token. Faster and more stable than gradient-based.
    """

    def __init__(self, model_name="distilgpt2", max_new_tokens=16, device="auto", model_obj=None, tokenizer_obj=None):
        if model_obj is not None and tokenizer_obj is not None:
            self.model = model_obj
            self.tokenizer = tokenizer_obj
            self.device = next(self.model.parameters()).device
        else:
            self.model, self.tokenizer = get_causal_model_and_tokenizer(model_name, device)
            self.device = next(self.model.parameters()).device
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def calculate(self, text: str, words):
        """Attention-based importance: single forward pass, no autograd."""
        if not words:
            return []

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        input_ids = encoded["input_ids"].to(self.device)
        offsets = encoded["offset_mapping"].squeeze(0).tolist()

        # Single forward pass with attention output
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_attentions=True,
        )

        # Last layer attention: (1, num_heads, seq_len, seq_len)
        last_attn = outputs.attentions[-1]

        # Average across all attention heads: (seq_len, seq_len)
        avg_attn = last_attn.squeeze(0).mean(dim=0)

        # Token importance = how much attention each token RECEIVES from others
        # Sum columns: each token's received attention from all positions
        token_importance = avg_attn.sum(dim=0).cpu()

        # Map token-level importance to word-level
        raw_scores = []
        for word in words:
            selected = []
            for i, (start, end) in enumerate(offsets):
                if end <= word["start"] or start >= word["end"]:
                    continue
                if i < len(token_importance):
                    selected.append(float(token_importance[i]))

            raw_scores.append(
                sum(selected) / len(selected) if selected else 0.0
            )

        return normalize(raw_scores)

    def calculate_gradient(self, text: str, words):
        """Gradient-based importance (original ProSan method). Slower but available for research."""
        if not words:
            return []

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        input_ids = encoded["input_ids"].to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = input_ids.shape[1]

        if generated.shape[1] <= prompt_len:
            return [0.0 for _ in words]

        full_ids = generated
        embedding_layer = self.model.get_input_embeddings()

        prompt_embeds = embedding_layer(input_ids).detach()
        prompt_embeds.requires_grad_(True)

        continuation_ids = full_ids[:, prompt_len:]
        continuation_embeds = embedding_layer(continuation_ids).detach()

        full_embeds = torch.cat([prompt_embeds, continuation_embeds], dim=1)

        attention_mask = torch.ones(
            full_embeds.shape[:2],
            dtype=torch.long,
            device=self.device,
        )

        outputs = self.model(
            inputs_embeds=full_embeds,
            attention_mask=attention_mask,
        )

        logits = outputs.logits[:, prompt_len - 1:-1, :]
        labels = full_ids[:, prompt_len:]

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
        )

        self.model.zero_grad(set_to_none=True)
        loss.backward()

        token_grads = prompt_embeds.grad.detach().norm(dim=-1).squeeze(0).cpu()
        offsets = encoded["offset_mapping"].squeeze(0).tolist()

        raw_scores = []
        for word in words:
            selected = []
            for i, (start, end) in enumerate(offsets):
                if end <= word["start"] or start >= word["end"]:
                    continue
                selected.append(float(token_grads[i]))

            raw_scores.append(
                sum(selected) / len(selected) if selected else 0.0
            )

        return normalize(raw_scores)


def normalize(values):
    if not values:
        return []

    lo = min(values)
    hi = max(values)

    if hi == lo:
        return [0.0 for _ in values]

    return [(x - lo) / (hi - lo) for x in values]
