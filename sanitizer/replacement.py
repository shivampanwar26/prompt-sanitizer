import hashlib
import math
import random
from typing import Iterable, List, Optional, Set, Tuple

import torch

from .model_registry import get_mask_model_and_tokenizer, get_vocab_masks
from .selector import SelectionUnit, _STOPWORDS
from .similarity import SimilarityCalculator


class ReplacementGenerator:
    """
    ProSan Sanitized Word Generation (Section V-B) — Enhanced.

    1. Entity-level masking: *every* mention of the entity is masked in the
       same input and the per-position log-probabilities are averaged, so the
       MLM cannot simply copy the name back from another mention.
    2. Vocabulary masks restrict candidates to whole, alphabetic words with the
       original's capitalisation – no sub-word fragments, digits or punctuation.
    3. Leak filter: the original word, anything sharing its stem, stop words and
       every other sensitive word of the prompt are never proposed.
    4. Cumulative truncation at tau, then Eq. 9 re-scoring on the MLM prior:
           p'_i  ∝  p_i * exp((K_w - eta * O_w) * s_i)
       (s_i = embedding cosine similarity), so high-privacy words move away
       from look-alike candidates while staying fluent in context.
    5. All entities are scored in one batched forward pass; sampling uses a
       per-entity seeded RNG, so results are reproducible and a given entity
       gets the same replacement wherever it appears.
    """

    CONTEXT_CHARS = 1500  # window around the entity sent to the MLM (512-token limit)

    def __init__(self, model_name="roberta-base", top_k=15, eta=1.0, tau=0.9,
                 sampling_mode="sample", device="auto", seed=42):
        self.top_k = top_k
        self.eta = eta
        self.tau = tau
        self.sampling_mode = sampling_mode
        self.seed = seed

        self.model, self.tokenizer = get_mask_model_and_tokenizer(model_name, device)
        self.device = next(self.model.parameters()).device
        self.vocab_masks = get_vocab_masks(model_name, self.tokenizer, self.device)
        self.similarity_calc = SimilarityCalculator(model=self.model, tokenizer=self.tokenizer)

    # ── helpers ─────────────────────────────────────────────────
    def _masked_input(self, text: str, unit: SelectionUnit) -> Tuple[str, str]:
        """Mask every occurrence (inside a context window); return (masked text, vocab key)."""
        first_start, first_end = unit.occurrences[0]
        lo = max(0, first_start - self.CONTEXT_CHARS // 2)
        hi = min(len(text), first_end + self.CONTEXT_CHARS // 2)

        pieces, cursor = [], lo
        for start, end in sorted(unit.occurrences):
            if start < lo or end > hi:
                continue
            pieces.append(text[cursor:start])
            pieces.append(self.tokenizer.mask_token)
            cursor = end
        pieces.append(text[cursor:hi])

        prev = text[first_start - 1] if first_start > 0 else ""
        ws = "ws" if prev in (" ", "\t", "\n") else "nows"
        case = "upper" if unit.text[:1].isupper() else "lower"
        return "".join(pieces), f"{ws}_{case}"

    @staticmethod
    def _leaks(candidate: str, forbidden: Set[str]) -> bool:
        c = candidate.lower()
        if c in forbidden or c in _STOPWORDS:
            return True
        for f in forbidden:
            if len(f) >= 4 and len(c) >= 4 and (c[:4] == f[:4] or f in c or c in f):
                return True
        return False

    def _rng(self, unit: SelectionUnit) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}\x1f{unit.text.lower()}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    # ── candidate generation ────────────────────────────────────
    @torch.no_grad()
    def candidates(self, text: str, units: List[SelectionUnit],
                   forbidden: Iterable[str] = ()) -> List[List[Tuple[str, int, float]]]:
        """Return, for each unit, [(word, token_id, p_i)] after filtering and tau-truncation."""
        if not units:
            return []
        forbidden = {f.lower() for f in forbidden}

        inputs, keys = zip(*(self._masked_input(text, u) for u in units))
        encoded = self.tokenizer(list(inputs), return_tensors="pt", padding=True,
                                 truncation=True, max_length=512).to(self.device)
        logits = self.model(**encoded).logits

        results = []
        for i, unit in enumerate(units):
            positions = (encoded["input_ids"][i] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                results.append([])
                continue
            log_probs = torch.log_softmax(logits[i, positions].float(), dim=-1).mean(dim=0)
            log_probs = log_probs.masked_fill(~self.vocab_masks[keys[i]], float("-inf"))
            probs = torch.softmax(log_probs, dim=-1)

            unit_forbidden = forbidden | {p.lower() for p in unit.text.split()}
            top_p, top_i = torch.topk(probs, min(self.top_k * 4, probs.shape[-1]))
            picked = []
            for p, idx in zip(top_p.tolist(), top_i.tolist()):
                if p <= 0:
                    break
                word = self.tokenizer.decode([idx]).strip()
                if not word or self._leaks(word, unit_forbidden):
                    continue
                picked.append((word, idx, p))
                if len(picked) >= self.top_k:
                    break

            # Cumulative truncation (paper): smallest prefix with mass >= tau.
            total = sum(p for _, _, p in picked)
            kept, cum = [], 0.0
            for word, idx, p in picked:
                kept.append((word, idx, p / total))
                cum += p / total
                if cum >= self.tau:
                    break
            results.append(kept)
        return results

    def choose(self, unit: SelectionUnit, candidates: List[Tuple[str, int, float]]) -> Optional[str]:
        """Eq. 9 re-scoring on top of the MLM prior, then sample or argmax."""
        if not candidates:
            return None
        sims = self.similarity_calc.similarity_to_ids(unit.text, [idx for _, idx, _ in candidates])
        exponent = unit.importance - self.eta * unit.privacy
        scores = [p * math.exp(exponent * s) for (_, _, p), s in zip(candidates, sims)]
        total = sum(scores)
        words = [w for w, _, _ in candidates]

        # Degenerate scores (e.g. a zero embedding vector) must never crash
        # sampling; fall back to the masked LM's own ranking instead.
        if not math.isfinite(total) or total <= 0:
            return candidates[max(range(len(candidates)), key=lambda i: candidates[i][2])][0]
        weights = [s / total for s in scores]
        if self.sampling_mode == "max" or len(words) == 1:
            return words[max(range(len(words)), key=weights.__getitem__)]
        return self._rng(unit).choices(words, weights=weights, k=1)[0]

    def replace(self, text: str, units: List[SelectionUnit], forbidden: Iterable[str] = ()) -> None:
        """Fill ``unit.replacement`` for every unit (None when no safe candidate exists)."""
        # Numeric identifiers are better served by format-preserving surrogates.
        textual = [u for u in units if not any(ch.isdigit() for ch in u.text)]
        forbidden = set(forbidden) | {u.text for u in units} | {m.word for u in units for m in u.members}
        for unit, cands in zip(textual, self.candidates(text, textual, forbidden)):
            unit.replacement = self.choose(unit, cands) or ""
