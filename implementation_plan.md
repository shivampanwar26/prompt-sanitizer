# ProSan Prompt Privacy Sanitizer — Enhanced Implementation Plan

This plan details the full enhancement of the **Prompt Sanitizer** codebase to faithfully align with the paper *"ProSan: Utility-Based Prompt Privacy Sanitizer"* (IEEE TIFS 2026).

## User Review Required

> [!IMPORTANT]
> The current starter codebase relies on static thresholding and basic character intersection logic for word substitution. Upgrading to the paper's full mathematical model introduces dynamic self-information ratio calculation ($\gamma_q$), WordNet path similarity ($s_i$), cumulative probability truncation ($\tau$), softmax candidate probability recalculation ($p'_i$), NLTK POS tagging, repeated entity score max-pooling, and multi-turn context support.
> 
> Small default models (`distilgpt2` and `roberta-base`) will remain supported for CPU / fast low-resource execution, while configuration will allow seamlessly pointing to larger models (e.g., Llama-2 / Phi-3 / Gemma).

## Proposed Changes

### 1. Configuration & Domain Models

#### [NEW] [`config.py`](file:///c:/Users/panwa/OneDrive/Desktop/prompt-sanitizer/sanitizer/config.py)
- Create a `SanitizerConfig` dataclass and YAML loader to manage:
  - `causal_model` (default: `distilgpt2`)
  - `mask_model` (default: `roberta-base`)
  - `lambda_scale` (scaling factor $\lambda$ for dynamic protection ratio $\gamma_q$, default: `0.5`)
  - `min_privacy` (minimum self-information threshold)
  - `eta` ($\eta$ hyperparameter balancing utility vs privacy in Eq. 9, default: `1.0`)
  - `tau` (cumulative probability candidate set threshold $\tau$, default: `0.9`)
  - `top_k` (maximum candidate pool size for MLM)
  - `sampling_mode` (`"sample"` or `"max"`)
  - `device` (`"auto"`, `"cpu"`, `"cuda"`)
  - `pos_filter_enabled` (boolean, default: `True`)

#### [MODIFY] [`config/config.yaml`](file:///c:/Users/panwa/OneDrive/Desktop/prompt-sanitizer/config/config.yaml)
- Expand YAML parameters to include `lambda_scale`, `eta`, `tau`, `sampling_mode`, `pos_filter_enabled`, and `device`.

#### [MODIFY] [`types.py`](file:///c:/Users/panwa/OneDrive/Desktop/prompt-sanitizer/sanitizer/types.py)
- Enhance `WordScore` and `SanitizationResult` with additional metadata:
  - `pos_tag` (part-of-speech tag)
  - `H_q` (prompt average self-information score)
  - `gamma_q` (dynamically calculated protection ratio)
  - `candidate_info` (details on selected candidate word and probability)

---

### 2. Core Sanitizer Pipeline Engine

#### [MODIFY] [`selector.py`](file:///c:/Users/panwa/OneDrive/Desktop/prompt-sanitizer/sanitizer/selector.py)
- Implement dynamic protection ratio formula:
  $$\gamma_q = \lambda \cdot \frac{1}{1 + e^{-H_q}}$$
  where $H_q = \frac{1}{k} \sum_{j=1}^k I_{w_j}$ is the average self-information over all words in prompt $q$.
- Implement POS tagging filter using `nltk.pos_tag` (filtering out function words like prepositions, articles, conjunctions, pronouns while retaining content words: nouns, proper nouns, adjectives, adverbs, numbers).
- Sort candidates by ascending importance score $K_w$ to desensitize low-impact, high-privacy content words first.

#### [MODIFY] [`similarity.py`](file:///c:/Users/panwa/OneDrive/Desktop/prompt-sanitizer/sanitizer/similarity.py)
- Replace character-intersection dummy logic with WordNet synset path similarity ($s_i = \text{PathSimilarity}(w', c_i) \in [0, 1]$) using `nltk.corpus.wordnet`.
- Add robust fallback (Levenshtein / character / substring similarity) if WordNet synsets are unavailable or tokens are non-dictionary/code strings.

#### [MODIFY] [`replacement.py`](file:///c:/Users/panwa/OneDrive/Desktop/prompt-sanitizer/sanitizer/replacement.py)
- Implement cumulative probability thresholding ($\tau$) to truncate MLM top candidates:
  $$\sum_{i=1}^{m-1} p_i < \tau \le \sum_{i=1}^m p_i$$
- Implement exact sampling probability recalculation (Eq. 9):
  $$p'_i = \frac{e^{(K_w - \eta O_w) \cdot s_i}}{\sum_{j=1}^m e^{(K_w - \eta O_w) \cdot s_j}}$$
- Implement weighted stochastic sampling $c \sim p'_i$ or greedy deterministic selection.

#### [MODIFY] [`sanitizer.py`](file:///c:/Users/panwa/OneDrive/Desktop/prompt-sanitizer/sanitizer/sanitizer.py)
- Support repeated entity max-pooling: when the same word occurs multiple times in a prompt, assign the max privacy score across all occurrences to prevent decay of self-information on later mentions (Section VII-A).
- Add support for multi-turn prompt sanitization (Section VII-B): accept optional `history: List[str]` to construct complete context for importance/privacy evaluation.
- Support runtime config overrides in `sanitize(prompt, config_overrides=...)`.

#### [NEW] [`evaluator.py`](file:///c:/Users/panwa/OneDrive/Desktop/prompt-sanitizer/sanitizer/evaluator.py)
- Add utility to compute text perplexity (using the causal LM) for readability verification (Section VI-B3).
- Add evaluation helper to measure Privacy Hiding Rate (PHR) against target entity lists.

---

### 3. Application Interfaces & Tests

#### [MODIFY] [`app/main.py`](file:///c:/Users/panwa/OneDrive/Desktop/prompt-sanitizer/app/main.py)
- Upgrade FastAPI app:
  - `/sanitize`: accept optional context history and parameters (`lambda_scale`, `eta`, `tau`, `sampling_mode`). Return detailed response with $H_q$, $\gamma_q$, word scores, and perplexity metrics.
  - `/evaluate`: endpoint to compute original vs. sanitized prompt perplexity and privacy reduction.
  - `/config`: view active sanitizer settings.

#### [MODIFY] [`app/cli.py`](file:///c:/Users/panwa/OneDrive/Desktop/prompt-sanitizer/app/cli.py)
- Add CLI command-line arguments using `argparse`:
  - `--file` / `--text` for single or batch processing
  - `--eval` to output perplexity and privacy stats
  - `--history` for multi-turn context
  - `--config` to override YAML settings

#### [MODIFY] [`tests/test_tokenizer.py`](file:///c:/Users/panwa/OneDrive/Desktop/prompt-sanitizer/tests/test_tokenizer.py) & [NEW] [`tests/test_sanitizer.py`](file:///c:/Users/panwa/OneDrive/Desktop/prompt-sanitizer/tests/test_sanitizer.py)
- Comprehensive test suite testing:
  1. Word extraction and span mapping
  2. Importance calculation and normalization
  3. Self-information calculation & repeated entity max pooling
  4. WordNet path similarity & fallback
  5. Dynamic protection ratio calculation ($\gamma_q$) & POS filtering
  6. Cumulative probability thresholding ($\tau$) & candidate softmax recalculation ($p'_i$)
  7. Full end-to-end prompt sanitization pipeline
  8. FastAPI `/sanitize` and `/evaluate` REST endpoints

---

## Verification Plan

### Automated Tests
- Run `pytest` to execute unit tests for all components:
  ```bash
  pytest -v
  ```

### Manual Verification
- Execute CLI with test prompts (e.g. medical QA, SAMSum summary, CodeAlpaca prompts):
  ```bash
  python -m app.cli "Patient John Doe in New York has severe chest pain and dyspnea."
  ```
- Start uvicorn server and verify API endpoints:
  ```bash
  uvicorn app.main:app --port 8000
  ```
