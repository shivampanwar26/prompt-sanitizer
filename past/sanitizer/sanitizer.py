import re
from dataclasses import replace
from typing import List, Optional, Dict, Any

from .config import SanitizerConfig
from .importance import ImportanceCalculator
from .privacy import PrivacyCalculator
from .replacement import ReplacementGenerator
from .selector import WordSelector
from .tokenizer import extract_words
from .types import SanitizationResult, WordScore
from .pii import find_pii, get_fake_replacement
from .model_registry import get_causal_model_and_tokenizer
from .evaluator import PromptEvaluator


class PromptSanitizer:
    """
    ProSan Prompt Privacy Sanitizer (IEEE TIFS 2026).
    """

    def __init__(self, config: Optional[SanitizerConfig] = None):
        self.config = config or SanitizerConfig.load_from_yaml()

        self.importance = None
        self.privacy = None
        self.selector = None
        self.replacement = None
        self.evaluator = None
        self._load_prosan_components()

    def _load_prosan_components(self) -> None:
        if self.importance is not None:
            return

        causal_model_obj, causal_tokenizer_obj = get_causal_model_and_tokenizer(
            self.config.causal_model, self.config.device
        )

        self.importance = ImportanceCalculator(
            model_name=self.config.causal_model,
            device=self.config.device,
            model_obj=causal_model_obj,
            tokenizer_obj=causal_tokenizer_obj,
        )

        self.privacy = PrivacyCalculator(
            model_name=self.config.causal_model,
            device=self.config.device,
            model_obj=causal_model_obj,
            tokenizer_obj=causal_tokenizer_obj,
        )

        self.evaluator = PromptEvaluator(
            model_name=self.config.causal_model,
            device=self.config.device,
            model_obj=causal_model_obj,
            tokenizer_obj=causal_tokenizer_obj,
        )

        self.selector = WordSelector(
            self.config.lambda_scale,
            self.config.min_privacy,
            self.config.max_importance,
            self.config.pos_filter_enabled,
        )

        self.replacement = ReplacementGenerator(
            model_name=self.config.mask_model,
            top_k=self.config.top_k,
            eta=self.config.eta,
            tau=self.config.tau,
            sampling_mode=self.config.sampling_mode,
            device=self.config.device,
            seed=self.config.seed,
        )

    def _redact_pii_spans(self, prompt: str) -> (str, List[Dict[str, Any]]):
        spans = find_pii(prompt)
        if not spans:
            return prompt, []

        text = prompt
        selected = []
        for span in reversed(spans):
            original = prompt[span.start:span.end]
            # Use realistic fake value instead of raw <TAG> placeholder
            replacement = get_fake_replacement(span.kind, seed_text=original)
            text = text[:span.start] + replacement + text[span.end:]
            selected.append({
                "word": original,
                "start": span.start,
                "end": span.end,
                "importance": 0.0,
                "privacy": 1.0,
                "raw_privacy": 12.0,
                "pos_tag": f"PII:{span.kind}",
                "replacement": replacement,
            })
        selected.reverse()
        return text, selected

    def sanitize(
        self,
        prompt: str,
        history: Optional[List[str]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        evaluate: bool = True,
    ) -> SanitizationResult:
        if not prompt or not prompt.strip():
            return SanitizationResult(
                text=prompt,
                selected_words=[],
                H_q=0.0,
                gamma_q=0.0,
            )

        if config_overrides:
            unknown = set(config_overrides) - set(self.config.__dataclass_fields__)
            if unknown:
                raise ValueError(f"Unknown configuration override(s): {', '.join(sorted(unknown))}")
            effective_config = replace(self.config, **config_overrides)
        else:
            effective_config = self.config

        self._load_prosan_components()
        self.selector.lambda_scale = effective_config.lambda_scale
        self.selector.min_privacy = effective_config.min_privacy
        self.selector.max_importance = effective_config.max_importance
        self.selector.pos_filter_enabled = effective_config.pos_filter_enabled
        self.replacement.eta = effective_config.eta
        self.replacement.tau = effective_config.tau
        self.replacement.sampling_mode = effective_config.sampling_mode

        # Step 1: PII Pattern Redaction
        current_text, pii_selected = self._redact_pii_spans(prompt)

        # Step 2: ProSan Contextual Model Desensitization
        raw_words = extract_words(current_text)

        # Build ranges of fake-value replacements in the current (post-PII) text
        # so the model pipeline doesn't re-process them.
        pii_replacement_ranges = []
        offset = 0
        for item in pii_selected:
            orig_len = item["end"] - item["start"]
            repl_len = len(item["replacement"])
            new_start = item["start"] + offset
            new_end = new_start + repl_len
            pii_replacement_ranges.append((new_start, new_end))
            offset += repl_len - orig_len

        def is_in_pii_replacement(w):
            for p_start, p_end in pii_replacement_ranges:
                if w["start"] >= p_start and w["end"] <= p_end:
                    return True
            return False

        words = [w for w in raw_words if not is_in_pii_replacement(w)]

        orig_ppl = None  # Computed at end via batched call

        if not words:
            ppls = self.evaluator.perplexity_batch([prompt, current_text]) if self.evaluator and evaluate else [None, None]
            return SanitizationResult(
                text=current_text,
                selected_words=pii_selected,
                H_q=0.0,
                gamma_q=0.0,
                perplexity=ppls[1],
                original_perplexity=ppls[0],
            )

        # Section VII-B: Multi-turn prompt context
        if history:
            full_context = "\n".join(history) + "\n" + current_text
            offset = len("\n".join(history)) + 1
            context_words = extract_words(full_context)
            eval_text = full_context
            eval_words = [w for w in context_words if w["start"] >= offset and not is_in_pii_replacement({"start": w["start"] - offset, "end": w["end"] - offset})]
            importance_scores = self.importance.calculate(eval_text, eval_words)
            privacy_norm, privacy_raw = self.privacy.calculate(eval_text, eval_words)
        else:
            eval_text = current_text
            eval_words = words
            importance_scores = self.importance.calculate(eval_text, eval_words)
            privacy_norm, privacy_raw = self.privacy.calculate(eval_text, eval_words)

        selected_scores, H_q, gamma_q = self.selector.select(
            words,
            importance_scores,
            privacy_norm,
            privacy_raw,
            full_text=current_text,
        )

        if not selected_scores:
            ppls = self.evaluator.perplexity_batch([prompt, current_text]) if self.evaluator and evaluate else [None, None]
            return SanitizationResult(
                text=current_text,
                selected_words=pii_selected,
                H_q=round(H_q, 4),
                gamma_q=round(gamma_q, 4),
                perplexity=ppls[1],
                original_perplexity=ppls[0],
            )

        sanitized_text, desensitized_words = self.replacement.replace(
            current_text,
            selected_scores,
        )

        prosan_selected = [
            {
                "word": x.word,
                "start": x.start,
                "end": x.end,
                "importance": round(x.importance, 4),
                "privacy": round(x.privacy, 4),
                "raw_privacy": round(x.raw_privacy, 4),
                "pos_tag": x.pos_tag,
                "replacement": x.replacement,
            }
            for x in desensitized_words
        ]

        all_selected = pii_selected + prosan_selected

        # Batched evaluation: original + sanitized perplexity in one forward pass
        phr_val = None
        orig_ppl = None
        san_ppl = None
        if evaluate and self.evaluator:
            ppls = self.evaluator.perplexity_batch([prompt, sanitized_text])
            orig_ppl = ppls[0]
            san_ppl = ppls[1]
            orig_words = [item["word"] for item in all_selected if item["word"] not in sanitized_text]
            phr_info = PromptEvaluator.calculate_phr(orig_words, sanitized_text)
            phr_val = phr_info["phr"]

        return SanitizationResult(
            text=sanitized_text,
            selected_words=all_selected,
            H_q=round(H_q, 4),
            gamma_q=round(gamma_q, 4),
            perplexity=san_ppl,
            original_perplexity=orig_ppl,
            phr=phr_val,
        )
