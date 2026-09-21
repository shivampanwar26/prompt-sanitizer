import math
from typing import List, Optional

import torch


class SimilarityCalculator:
    """
    Embedding cosine similarity using the masked LM's own input embeddings.

    Replaces WordNet path_similarity with a faster, higher-coverage approach
    that works for any word including names, slang, code terms and
    abbreviations (WordNet has no synsets for most of what gets replaced).
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self._embedding_layer = None

    def _get_embedding_layer(self):
        if self._embedding_layer is None and self.model is not None and hasattr(self.model, "get_input_embeddings"):
            self._embedding_layer = self.model.get_input_embeddings()
        return self._embedding_layer

    @torch.no_grad()
    def _word_vector(self, word: str) -> Optional[torch.Tensor]:
        embed = self._get_embedding_layer()
        if embed is None or self.tokenizer is None:
            return None
        ids = self.tokenizer.encode(" " + word, add_special_tokens=False)
        if not ids:
            return None
        return embed(torch.tensor(ids, device=embed.weight.device)).float().mean(dim=0)

    @torch.no_grad()
    def similarity_to_ids(self, word: str, token_ids: List[int]) -> List[float]:
        """Cosine similarity (clamped to [0, 1]) between ``word`` and each vocabulary token."""
        vec = self._word_vector(word)
        if vec is None:
            return [0.0 for _ in token_ids]
        embed = self._get_embedding_layer()
        cands = embed(torch.tensor(token_ids, device=embed.weight.device)).float()
        sims = torch.nn.functional.cosine_similarity(cands, vec.unsqueeze(0), dim=-1)
        return sims.clamp(0.0, 1.0).tolist()

    def embedding_similarity(self, word1: str, word2: str) -> Optional[float]:
        v1, v2 = self._word_vector(word1), self._word_vector(word2)
        if v1 is None or v2 is None:
            return None
        return max(0.0, torch.nn.functional.cosine_similarity(v1, v2, dim=0).item())

    def fallback_similarity(self, original: str, candidate: str) -> float:
        """Character-overlap fallback when no embedding model is available."""
        w1, w2 = original.lower(), candidate.lower()
        if w1 == w2:
            return 1.0
        a, b = set(w1), set(w2)
        if not a or not b:
            return 0.0
        char_sim = len(a & b) / math.sqrt(len(a) * len(b))
        prefix = 0
        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                break
            prefix += 1
        return 0.7 * char_sim + 0.3 * prefix / max(len(w1), len(w2))

    def similarity(self, original: str, candidate: str) -> float:
        if original.lower() == candidate.lower():
            return 1.0
        emb = self.embedding_similarity(original, candidate)
        return emb if emb is not None else self.fallback_similarity(original, candidate)
