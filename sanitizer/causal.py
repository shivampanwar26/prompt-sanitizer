"""One causal-LM pass that yields both ProSan signals per word.

* Self-information  I(w) = -log2 P(w | context)   (privacy, Section IV-B)
* Utility importance K_w                          (Section IV-A)

Compared with computing the two in separate passes (and without a BOS token):

* A BOS token is prepended, so the first word gets a real surprisal instead
  of 0 – otherwise "Jack was born on ..." can never select "Jack".
* The BOS token also absorbs the *attention sink*.  Raw column sums of a
  causal attention matrix are dominated by the first position and are biased
  towards early tokens (they are attended by more rows).  Importance here is
  the attention a token receives divided by what it would receive under
  uniform attention, averaged over all layers and heads, with BOS excluded.
* Text longer than the model window is processed in windows, so long
  prompts are scored everywhere rather than silently truncated.
* ``importance_method="gradient"`` keeps the paper's gradient saliency
  (as gradient x input over a greedy continuation) for research use.
"""

import math
from typing import Dict, List, Sequence, Tuple

import torch

from .model_registry import get_causal_model_and_tokenizer

_LN2 = math.log(2.0)


def normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0 for _ in values]
    return [(x - lo) / (hi - lo) for x in values]


class CausalAnalyzer:
    def __init__(self, model_name="distilgpt2", device="auto", importance_method="attention",
                 model_obj=None, tokenizer_obj=None, max_new_tokens=16):
        if model_obj is not None and tokenizer_obj is not None:
            self.model, self.tokenizer = model_obj, tokenizer_obj
        else:
            self.model, self.tokenizer = get_causal_model_and_tokenizer(model_name, device)
        self.device = next(self.model.parameters()).device
        self.importance_method = importance_method
        self.max_new_tokens = max_new_tokens

        cfg = self.model.config
        self.window = int(getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", 1024)) - 1
        bos = self.tokenizer.bos_token_id
        self.bos_id = bos if bos is not None else self.tokenizer.eos_token_id

    # ── token level ─────────────────────────────────────────────
    @torch.no_grad()
    def _window_stats(self, ids: List[int]) -> Tuple[List[float], List[float]]:
        """Surprisal (bits) and debiased attention importance for one window."""
        input_ids = torch.tensor([[self.bos_id] + ids], device=self.device)
        want_attn = self.importance_method == "attention"
        out = self.model(input_ids=input_ids, output_attentions=want_attn)

        log_probs = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
        bits = (-log_probs.gather(1, input_ids[0, 1:].unsqueeze(-1)).squeeze(-1) / _LN2).tolist()

        if not want_attn or not out.attentions:
            return bits, [0.0] * len(ids)

        attn = torch.stack([a[0].float() for a in out.attentions]).mean(dim=(0, 1))  # (L, L)
        length = attn.shape[0]
        received = attn.sum(dim=0)                                   # column sums
        uniform = 1.0 / torch.arange(1, length + 1, device=attn.device, dtype=attn.dtype)
        expected = torch.flip(torch.cumsum(torch.flip(uniform, [0]), 0), [0])  # sum_{i>=j} 1/(i+1)
        ratio = (received / expected.clamp_min(1e-6))[1:]           # drop BOS
        return bits, ratio.tolist()

    def _gradient_importance(self, ids: List[int]) -> List[float]:
        input_ids = torch.tensor([[self.bos_id] + ids], device=self.device)
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
            return [0.0] * len(ids)

        embed = self.model.get_input_embeddings()
        prompt_embeds = embed(input_ids).detach().requires_grad_(True)
        full = torch.cat([prompt_embeds, embed(generated[:, prompt_len:]).detach()], dim=1)
        logits = self.model(inputs_embeds=full).logits[:, prompt_len - 1:-1, :]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), generated[:, prompt_len:].reshape(-1)
        )
        self.model.zero_grad(set_to_none=True)
        loss.backward()
        saliency = (prompt_embeds.grad * prompt_embeds).sum(-1).abs()[0, 1:]
        return saliency.detach().float().cpu().tolist()

    def token_stats(self, text: str):
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        bits, importance = [], []
        for start in range(0, len(ids), self.window):
            chunk = ids[start:start + self.window]
            b, imp = self._window_stats(chunk)
            if self.importance_method == "gradient":
                imp = self._gradient_importance(chunk)
            bits += b
            importance += imp
        return bits, importance, offsets

    # ── word level ──────────────────────────────────────────────
    def word_scores(self, text: str, words: List[Dict], offset: int = 0) -> Tuple[List[float], List[float]]:
        """Return (importance K_w in [0, 1], self-information bits) for each word.

        ``offset`` is where the words' coordinates begin inside ``text`` (used
        when ``text`` is prefixed with conversation history).
        """
        if not words:
            return [], []
        bits, importance, offsets = self.token_stats(text)

        word_bits, word_imp = [], []
        t = 0
        for word in words:
            ws, we = word["start"] + offset, word["end"] + offset
            while t < len(offsets) and offsets[t][1] <= ws:
                t += 1
            b, imp, k = 0.0, 0.0, t
            while k < len(offsets) and offsets[k][0] < we:
                if offsets[k][1] > ws:
                    b += bits[k]
                    imp = max(imp, importance[k])
                k += 1
            word_bits.append(b)
            word_imp.append(imp)
        return normalize(word_imp), word_bits
