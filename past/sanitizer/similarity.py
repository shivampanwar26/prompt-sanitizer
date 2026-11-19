import torch
from typing import Optional


class SimilarityCalculator:
    """
    Embedding cosine similarity using the masked LM's own embeddings.

    Replaces WordNet path_similarity with a faster, higher-coverage approach
    that works for any word including slang, code terms, and abbreviations.
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self._embedding_layer = None

    def _get_embedding_layer(self):
        if self._embedding_layer is None and self.model is not None:
            # Get the input embedding layer from the model
            if hasattr(self.model, "get_input_embeddings"):
                self._embedding_layer = self.model.get_input_embeddings()
            elif hasattr(self.model, "roberta"):
                self._embedding_layer = self.model.roberta.embeddings.word_embeddings
            elif hasattr(self.model, "bert"):
                self._embedding_layer = self.model.bert.embeddings.word_embeddings
        return self._embedding_layer

    @torch.no_grad()
    def embedding_similarity(self, word1: str, word2: str) -> Optional[float]:
        """Cosine similarity between word embeddings from the masked LM."""
        embed_layer = self._get_embedding_layer()
        if embed_layer is None or self.tokenizer is None:
            return None

        try:
            ids1 = self.tokenizer.encode(word1, add_special_tokens=False)
            ids2 = self.tokenizer.encode(word2, add_special_tokens=False)

            if not ids1 or not ids2:
                return None

            device = embed_layer.weight.device
            # Average embedding across sub-word tokens
            emb1 = embed_layer(torch.tensor(ids1, device=device)).mean(dim=0)
            emb2 = embed_layer(torch.tensor(ids2, device=device)).mean(dim=0)

            cos_sim = torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0).float(), emb2.unsqueeze(0).float()
            ).item()

            # Clamp to [0, 1] since we want non-negative similarity
            return max(0.0, cos_sim)
        except Exception:
            return None

    def fallback_similarity(self, original: str, candidate: str) -> float:
        """Character-overlap fallback for edge cases."""
        if original.lower() == candidate.lower():
            return 1.0

        a = set(original.lower())
        b = set(candidate.lower())

        if not a or not b:
            return 0.0

        import math
        char_sim = len(a & b) / math.sqrt(len(a) * len(b))

        w1, w2 = original.lower(), candidate.lower()
        common_prefix = 0
        for c1, c2 in zip(w1, w2):
            if c1 == c2:
                common_prefix += 1
            else:
                break
        prefix_sim = common_prefix / max(len(w1), len(w2))

        return 0.7 * char_sim + 0.3 * prefix_sim

    def similarity(self, original: str, candidate: str) -> float:
        if original.lower() == candidate.lower():
            return 1.0

        emb_sim = self.embedding_similarity(original, candidate)
        if emb_sim is not None:
            return emb_sim

        return self.fallback_similarity(original, candidate)
