import re
from typing import Any, Dict, List, Optional

from .config import SanitizerConfig
from .nltk_resources import pos_tag
from .pii import find_pii
from .privacy import PrivacyCalculator
from .selector import WordSelector
from .surrogates import SurrogateGenerator, match_case
from .tokenizer import extract_words
from .types import SanitizationResult

_ENTITY_KINDS = ("PERSON", "LOCATION", "ORGANIZATION")


class SanitizationSession:
    """State shared by the turns of one conversation.

    Keeps surrogates consistent (the same person is "Alex" in every turn) and
    remembers what was hidden, so an entity sanitized in turn 1 is also hidden
    in turn 5 even when turn 5 alone would not flag it.
    """

    def __init__(self, style: str = "realistic", seed: Optional[int] = 42):
        self.surrogates = SurrogateGenerator(style=style, secret=f"prosan-{seed}")
        self.contextual: Dict[str, str] = {}   # lower-cased original -> replacement
        self.history: List[str] = []

    @property
    def mapping(self) -> Dict[str, str]:
        return self.surrogates.mapping


class PromptSanitizer:
    """
    ProSan Prompt Privacy Sanitizer (IEEE TIFS 2026) — Enhanced ("ProSan+").

    Pipeline (all offsets refer to the original prompt):
      1. Validated recognizers + context rules + NER find explicit PII and secrets.
      2. One causal-LM pass scores every remaining word for utility (K_w) and
         self-information; a specificity prior turns that into calibrated risk O_w.
      3. Entity-level adaptive selection under the gamma_q budget.
      4. Masked-LM replacement with Eq. 9 re-scoring (or placeholders).
      5. Consistent surrogates for everything, applied in a single pass,
         with a mapping that can restore the LLM's answer.
    """

    def __init__(self, config: Optional[SanitizerConfig] = None):
        self.config = config or SanitizerConfig.load_from_yaml()
        self._analyzer = None
        self._replacer = None
        self._evaluator = None
        self.selector = WordSelector()
        self.privacy = PrivacyCalculator()
        if self.config.mode == "prosan":
            self._load_prosan_components()

    # ── lazy components ─────────────────────────────────────────
    def _load_prosan_components(self) -> None:
        if self._analyzer is not None:
            return
        from .causal import CausalAnalyzer
        from .model_registry import get_causal_model_and_tokenizer

        model, tokenizer = get_causal_model_and_tokenizer(self.config.causal_model, self.config.device)
        self._analyzer = CausalAnalyzer(model_obj=model, tokenizer_obj=tokenizer,
                                        importance_method=self.config.importance_method)

    def _get_replacer(self, cfg: SanitizerConfig):
        if self._replacer is None:
            from .replacement import ReplacementGenerator

            self._replacer = ReplacementGenerator(model_name=cfg.mask_model, device=cfg.device, seed=cfg.seed)
        self._replacer.top_k, self._replacer.eta, self._replacer.tau = cfg.top_k, cfg.eta, cfg.tau
        self._replacer.sampling_mode, self._replacer.seed = cfg.sampling_mode, cfg.seed
        return self._replacer

    @property
    def evaluator(self):
        if self._evaluator is None:
            from .evaluator import PromptEvaluator

            self._evaluator = PromptEvaluator(model_name=self.config.causal_model, device=self.config.device)
        return self._evaluator

    def new_session(self, config: Optional[SanitizerConfig] = None) -> SanitizationSession:
        cfg = config or self.config
        return SanitizationSession(style=cfg.surrogate_style, seed=cfg.seed)

    # ── main entry point ────────────────────────────────────────
    def sanitize(
        self,
        prompt: str,
        history: Optional[List[str]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        evaluate: bool = True,
        session: Optional[SanitizationSession] = None,
    ) -> SanitizationResult:
        if not prompt or not prompt.strip():
            return SanitizationResult(text=prompt, selected_words=[])

        cfg = self.config.with_overrides(config_overrides) if config_overrides else self.config
        session = session or self.new_session(cfg)
        history = list(history or session.history)

        # Step 1: explicit PII, secrets and named entities.
        spans = find_pii(
            prompt, ner_backend=cfg.ner_backend, ner_model=cfg.ner_model,
            ner_threshold=cfg.ner_threshold, ner_labels=cfg.ner_labels,
            propagate=cfg.propagate_entities, device=cfg.device,
        )
        edits: List[Dict[str, Any]] = []
        # Longest mentions first, so "Rahul" inherits the surrogate of "Rahul Sharma".
        for span in sorted(spans, key=lambda s: -len(prompt[s.start:s.end].split())):
            original = prompt[span.start:span.end]
            edits.append(self._edit(span.start, span.end, original,
                                    session.surrogates.get(span.kind, original),
                                    pos_tag=f"PII:{span.kind}", source=span.source))

        def free(start: int, end: int) -> bool:
            return all(end <= e["start"] or start >= e["end"] for e in edits)

        words = [w for w in extract_words(prompt) if free(w["start"], w["end"])]

        # Entities hidden in earlier turns stay hidden, with the same surrogate.
        remembered = []
        for w in words:
            key = w["word"].lower()
            earlier = session.contextual.get(key)
            if earlier is None:
                earlier = next((session.surrogates.lookup(k, w["word"]) for k in _ENTITY_KINDS
                                if session.surrogates.lookup(k, w["word"])), None)
            if earlier is not None:
                remembered.append(w)
                edits.append(self._edit(w["start"], w["end"], w["word"], match_case(w["word"], earlier),
                                        pos_tag="MEMORY", source="session"))
        words = [w for w in words if w not in remembered]

        H_q = gamma_q = 0.0
        if cfg.mode == "prosan" and words:
            H_q, gamma_q = self._contextual(prompt, words, history, cfg, session, edits)

        # Step 5: apply every edit right-to-left on the original prompt.
        edits.sort(key=lambda e: e["start"])
        text = prompt
        for edit in reversed(edits):
            text = text[:edit["start"]] + edit["replacement"] + text[edit["end"]:]
        session.history = history + [text]

        result = SanitizationResult(
            text=text, selected_words=edits, H_q=round(H_q, 4), gamma_q=round(gamma_q, 4),
            mapping=session.mapping,
        )
        if evaluate:
            from .evaluator import PromptEvaluator

            result.original_perplexity, result.perplexity = self.evaluator.perplexity_batch([prompt, text])
            result.phr = PromptEvaluator.calculate_phr([e["word"] for e in edits], text)["phr"]
        return result

    def _contextual(self, prompt, words, history, cfg, session, edits):
        """Steps 2-4: score, select and replace words that no recognizer flagged."""
        self._load_prosan_components()
        self._analyzer.importance_method = cfg.importance_method

        prefix = ("\n".join(history) + "\n") if history else ""
        importance, bits = self._analyzer.word_scores(prefix + prompt, words, offset=len(prefix))
        tags = pos_tag([w["word"] for w in words])

        self.privacy.saturation = cfg.privacy_saturation
        privacy, raw = self.privacy.calculate(prompt, words, bits, tags)

        self.selector.lambda_scale, self.selector.min_privacy = cfg.lambda_scale, cfg.min_privacy
        self.selector.max_importance, self.selector.pos_filter_enabled = cfg.max_importance, cfg.pos_filter_enabled
        units, H_q, gamma_q = self.selector.select(words, importance, privacy, raw, tags, full_text=prompt)
        if not units:
            return H_q, gamma_q

        if cfg.surrogate_style == "realistic":
            forbidden = [e["word"] for e in edits]
            self._get_replacer(cfg).replace(prompt, units, forbidden)

        for unit in units:
            replacement = unit.replacement
            if not replacement or cfg.surrogate_style == "placeholder":
                kind = ("ID_NUMBER" if any(c.isdigit() for c in unit.text)
                        else "PERSON" if unit.text[:1].isupper() and cfg.surrogate_style == "realistic"
                        else "SENSITIVE")
                replacement = session.surrogates.get(kind, unit.text)
            else:
                session.surrogates.register("CONTEXTUAL", unit.text, replacement)
            session.contextual[unit.text.lower()] = replacement

            for start, end in unit.occurrences:
                original = prompt[start:end]
                members = [m for m in unit.members if start <= m.start < end]
                edits.append(self._edit(
                    start, end, original, match_case(original, replacement),
                    importance=max(m.importance for m in members),
                    privacy=max(m.privacy for m in members),
                    raw_privacy=sum(m.raw_privacy for m in members),
                    pos_tag=members[0].pos_tag, source="contextual",
                ))
        return H_q, gamma_q

    @staticmethod
    def _edit(start, end, word, replacement, importance=0.0, privacy=1.0, raw_privacy=0.0,
              pos_tag="", source="pattern") -> Dict[str, Any]:
        return {
            "word": word, "start": start, "end": end,
            "importance": round(importance, 4), "privacy": round(privacy, 4),
            "raw_privacy": round(raw_privacy, 4), "pos_tag": pos_tag,
            "replacement": replacement, "source": source,
        }

    def sanitize_batch(self, prompts: List[str], **kwargs) -> List[SanitizationResult]:
        return [self.sanitize(p, **kwargs) for p in prompts]

    # ── de-anonymization ────────────────────────────────────────
    @staticmethod
    def restore(text: str, mapping: Dict[str, str]) -> str:
        """Map surrogates in an LLM response back to the original values."""
        if not mapping or not text:
            return text
        keys = sorted(mapping, key=len, reverse=True)
        pattern = re.compile("|".join(
            (r"(?<!\w)" if k[:1].isalnum() else "") + re.escape(k) + (r"(?!\w)" if k[-1:].isalnum() else "")
            for k in keys
        ))
        return pattern.sub(lambda m: mapping[m.group(0)], text)
