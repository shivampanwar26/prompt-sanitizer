import math
import random
from typing import List, Tuple, Optional
import torch
from .model_registry import get_mask_model_and_tokenizer
from .similarity import SimilarityCalculator
from .types import WordScore


class ReplacementGenerator:
    """
    ProSan Sanitized Word Generation (Section V-B) — Enhanced.

    Improvements over the paper:
    1. Batched replacement: all mask positions in one forward pass
    2. Contextual re-scoring: MLM probability × ProSan score
    3. Embedding similarity: cosine similarity from model embeddings
    """

    def __init__(
        self,
        model_name="roberta-base",
        top_k=15,
        eta=1.0,
        tau=0.9,
        sampling_mode="sample",
        device="auto",
        seed=42,
    ):
        self.top_k = top_k
        self.eta = eta
        self.tau = tau
        self.sampling_mode = sampling_mode

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.model, self.tokenizer = get_mask_model_and_tokenizer(model_name, device)
        self.device = next(self.model.parameters()).device

        # Pass model/tokenizer to similarity calculator for embedding-based similarity
        self.similarity_calc = SimilarityCalculator(
            model=self.model, tokenizer=self.tokenizer
        )

    @torch.no_grad()
    def _batch_get_candidates(
        self, text: str, words: List[WordScore]
    ) -> List[List[Tuple[str, float]]]:
        """Get candidates for ALL words in a single batched forward pass."""
        if not words:
            return []

        mask_token = self.tokenizer.mask_token
        mask_token_id = self.tokenizer.mask_token_id

        # Create N masked copies of the text
        masked_texts = []
        for word in words:
            masked = text[:word.start] + mask_token + text[word.end:]
            masked_texts.append(masked)

        # Tokenize all copies with padding
        encoded = self.tokenizer(
            masked_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Single batched forward pass
        logits = self.model(**encoded).logits  # (N, seq_len, vocab)

        all_candidates = []
        for i in range(len(words)):
            # Find mask position in this sequence
            mask_positions = (
                encoded["input_ids"][i] == mask_token_id
            ).nonzero(as_tuple=True)[0]

            if len(mask_positions) == 0:
                all_candidates.append([])
                continue

            position = mask_positions[0]
            probs = torch.softmax(logits[i, position].float(), dim=-1)
            top_probs, top_indices = torch.topk(
                probs, min(self.top_k * 2, probs.shape[-1])
            )

            candidates = []
            cum_prob = 0.0
            for prob_val, idx in zip(top_probs.tolist(), top_indices.tolist()):
                candidate = self.tokenizer.decode(
                    [idx], skip_special_tokens=True
                ).strip()

                if not candidate or " " in candidate:
                    continue

                candidates.append((candidate, prob_val))
                cum_prob += prob_val

                if cum_prob >= self.tau or len(candidates) >= self.top_k:
                    break

            all_candidates.append(candidates)

        return all_candidates

    @torch.no_grad()
    def get_candidates_with_probabilities(
        self, text: str, word: WordScore
    ) -> List[Tuple[str, float]]:
        """Single-word fallback (used when batch fails)."""
        masked_text = (
            text[:word.start]
            + self.tokenizer.mask_token
            + text[word.end:]
        )

        encoded = self.tokenizer(masked_text, return_tensors="pt").to(self.device)

        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (
            encoded["input_ids"][0] == mask_token_id
        ).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            return []

        position = mask_positions[0]

        logits = self.model(**encoded).logits[0, position]
        probs = torch.softmax(logits.float(), dim=-1)

        top_probs, top_indices = torch.topk(probs, min(self.top_k * 2, probs.shape[-1]))

        candidates = []
        cum_prob = 0.0

        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            candidate = self.tokenizer.decode([idx], skip_special_tokens=True).strip()

            if not candidate or " " in candidate:
                continue

            candidates.append((candidate, prob))
            cum_prob += prob

            if cum_prob >= self.tau or len(candidates) >= self.top_k:
                break

        return candidates

    def _score_candidates(
        self, word: WordScore, candidates: List[Tuple[str, float]]
    ) -> str:
        """Score candidates using contextual re-scoring: P_MLM × exp(ProSan_score).

        This improves over the paper by incorporating the masked LM's own
        probability, ensuring replacements are both semantically appropriate
        AND privacy-maximizing.
        """
        if not candidates:
            return word.word  # No replacement found

        scored = []
        cand_words = []

        for cand, mlm_prob in candidates:
            s_i = self.similarity_calc.similarity(word.word, cand)
            # Contextual re-scoring: P_MLM(c_i) × exp((K_w - η·O_w) · s_i)
            prosan_score = (word.importance - self.eta * word.privacy) * s_i
            combined_score = mlm_prob * math.exp(prosan_score)
            scored.append(combined_score)
            cand_words.append(cand)

        # Normalize to probability distribution
        total = sum(scored)
        if total <= 0:
            return cand_words[0]
        p_prime = [s / total for s in scored]

        if self.sampling_mode == "max" or len(cand_words) == 1:
            best_idx = max(range(len(p_prime)), key=lambda i: p_prime[i])
            return cand_words[best_idx]
        else:
            return random.choices(cand_words, weights=p_prime, k=1)[0]

    def replace(
        self, text: str, selected_words: List[WordScore]
    ) -> Tuple[str, List[WordScore]]:
        """Replace selected words using batched inference + contextual re-scoring."""
        if not selected_words:
            return text, selected_words

        sorted_words = sorted(selected_words, key=lambda x: x.start, reverse=True)

        # Try batched replacement first
        try:
            # Get candidates for all words in one forward pass
            # Use forward-sorted order for batch, then apply in reverse
            forward_words = sorted(sorted_words, key=lambda x: x.start)
            all_candidates = self._batch_get_candidates(text, forward_words)

            # Map candidates back to reverse-sorted order
            candidate_map = {w.start: cands for w, cands in zip(forward_words, all_candidates)}

            result_text = text
            for word in sorted_words:
                candidates = candidate_map.get(word.start, [])
                replacement = self._score_candidates(word, candidates)
                word.replacement = replacement
                result_text = (
                    result_text[:word.start]
                    + replacement
                    + result_text[word.end:]
                )

        except Exception:
            # Fallback to sequential replacement
            result_text = text
            for word in sorted_words:
                candidates = self.get_candidates_with_probabilities(result_text, word)
                replacement = self._score_candidates(word, candidates)
                word.replacement = replacement
                result_text = (
                    result_text[:word.start]
                    + replacement
                    + result_text[word.end:]
                )

        return result_text, selected_words
