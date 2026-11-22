# ProSan+ — Utility-Based Prompt Privacy Sanitizer

Detects and desensitizes personal, secret and identifying information in a prompt before it
reaches an LLM — while keeping the words the task actually depends on. Inspired by
*"ProSan: Utility-Based Prompt Privacy Sanitizer"* (IEEE TIFS 2026), with a rewritten detection,
scoring and replacement pipeline (see [What changed](#what-changed-vs-the-paper--the-first-implementation) below).

```python
from sanitizer import PromptSanitizer

s = PromptSanitizer()
session = s.new_session()                                    # keeps surrogates consistent across turns
r = s.sanitize("My name is Rahul Sharma, call +91 98765 43210", session=session)
print(r.text)                                                 # -> "My name is Jordan Dalton, call +91 28683 61200"

answer = call_your_llm(r.text)
print(PromptSanitizer.restore(answer, r.mapping))             # real name/number back in the answer
```

## Pipeline

```
Original Prompt
  └─► Validated recognizers: email, phone, card (Luhn), IBAN (mod-97), SSN, Aadhaar (Verhoeff),
      PAN, IP/MAC, API keys & high-entropy secrets, credentials in code, dates, addresses, IDs
  └─► Context rules ("my name is X", "Dr. X", "I live in X") + transformer NER (PER/LOC/ORG)
  └─► Entity propagation: every mention of a detected entity is hidden
  └─► One causal-LM pass → self-information I_w and debiased attention importance K_w
  └─► Calibrated privacy risk O_w = specificity prior × (1 − e^(−I_w/κ)), repeated-entity max pooling
  └─► Entity-level adaptive selection: absolute threshold, capped by γ_q = λ·σ(H_q)
  └─► Masked LM over all mentions at once, leak filter, cumulative truncation (τ), Eq. 9 re-scoring
  └─► Consistent, format-preserving surrogates (dates shifted with intervals kept)
  └─► Sanitized Prompt + mapping (restore the LLM's answer with PromptSanitizer.restore)
```

Two independent layers, both always active in `mode: prosan`:

1. **Pattern/context/NER layer** (`sanitizer/pii.py`) — high-confidence, deterministic. Catches
   emails, phones, cards, IBANs, SSNs, Aadhaar, PAN, IPs/MACs, API keys, secrets and credentials in
   code, dates, addresses, IDs, and named people/places/organizations. Every mention of a caught
   entity is propagated and hidden, not just the first.
2. **Contextual layer** (`sanitizer/causal.py`, `privacy.py`, `selector.py`, `replacement.py`) —
   catches whatever the first layer misses, using a causal LM to score self-information and
   utility importance per word, an absolute calibrated privacy-risk score, and a masked LM to
   generate a fluent, non-identifying replacement.

Set `mode: pii_only` to run only the first layer — no language models are loaded, detection is
deterministic, and it's fast enough for CI or high-throughput pipelines.

## Project layout

```
sanitizer/          core library (see "Pipeline" above for the module-to-stage mapping)
  pii.py              recognizers, context rules, NER, entity propagation
  causal.py           shared causal-LM pass: self-information + attention importance
  privacy.py          calibrated privacy-risk scoring
  selector.py         entity-level adaptive word selection
  replacement.py      batched masked-LM replacement generation
  surrogates.py       deterministic, format-preserving fake values
  sanitizer.py        PromptSanitizer / SanitizationSession — the orchestrator
  evaluator.py        perplexity, PHR, utility-retention metrics
  config.py           SanitizerConfig (dataclass defaults + per-call overrides)
  image.py            ImageSanitizer: metadata, faces, OCR text (replace / box / blur / pixelate)
app/
  main.py             FastAPI app
  cli.py              command-line interface
frontend/            React/Vite dashboard (talks to app/main.py)
tests/               pytest suite (fast + @pytest.mark.models)
data/benchmark.jsonl gold-labelled prompts used by benchmark.py
data/testcases/      25 text + 25 image edge-case tests at none / medium / full privacy
run_image_cases.py   evaluates the image sanitizer on data/testcases/image_cases.jsonl
past/                the earlier implementation (incl. its seq2seq distillation), kept for reference
```

## Requirements & setup

Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

Models downloaded on first run (cached under `~/.cache/huggingface`):

| Model | Used for | Size |
|---|---|---|
| `distilgpt2` | self-information & importance scoring | ~330 MB |
| `roberta-base` | masked-LM replacement candidates | ~500 MB |
| `dslim/distilbert-NER` | named-entity recognition (set `ner_backend` to `nltk` or `none` to skip) | ~260 MB |
| RapidOCR (ONNX, in the pip wheel) | reading text in images when Tesseract isn't installed | ~16 MB |

NLTK resources (`wordnet`, `averaged_perceptron_tagger_eng`, `punkt_tab`, `maxent_ne_chunker_tab`,
`words`) also auto-download on first use, or pre-fetch them:

```bash
python -c "import nltk; [nltk.download(r, quiet=True) for r in ['wordnet','averaged_perceptron_tagger_eng','punkt_tab','maxent_ne_chunker_tab','words']]"
```

`mode="pii_only"` skips `distilgpt2`/`roberta-base` entirely; the NER model download is skipped
with `ner_backend="none"` (or `"nltk"`, which uses NLTK's bundled chunker instead). See
[Configuration](#configuration).

## Running the CLI

```bash
python -m app.cli "My name is John Doe and my email is john@example.com. I have chest pain."
```

| Flag | Purpose |
|---|---|
| `-f, --file PATH` | sanitize one prompt per line from a file (all lines share one session) |
| `--eval` | print perplexity, H_q, γ_q, and PHR (adds latency) |
| `--device {auto,cuda,cpu}` | Use CUDA automatically when available (default: `auto`) |
| `--image PATH [PATH ...]` | sanitize images instead of text (see [Image Sanitization](#image-sanitization)) |
| `--out-dir DIR` | where `--image` writes its output (default `sanitized_images`) |
| `--text-style {replace,box,blur,pixelate}` | how `--image` changes sensitive text (default `replace`) |
| `--no-face-redact`, `--no-text-redact`, `--deep` | image stage toggles; `--deep` adds the contextual layer |

```bash
python -m app.cli --eval "My name is John Doe and I live in Delhi."
python -m app.cli -f prompts.txt
python -m app.cli --image screenshot.png --text-style replace
```

Multi-turn history, `pii_only` mode and the other settings are available from Python and the
REST API (see below).

## Running the REST API

```bash
uvicorn app.main:app --reload --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness check |
| `GET /config` | current `SanitizerConfig` |
| `POST /sanitize` | sanitize a prompt; returns text, per-word detail, `H_q`/`γ_q`, and a `mapping` |
| `POST /sanitize/batch` | sanitize many prompts into (original, sanitized) pairs |
| `POST /sanitize/image` | sanitize an uploaded image; returns a PNG, report in `X-Sanitization-Report` |
| `POST /restore` | map surrogates in an LLM's reply back to the real values |
| `POST /evaluate` | perplexity and PHR for an (original, sanitized) pair you already have |

```bash
curl -X POST http://127.0.0.1:8000/sanitize \
  -H "Content-Type: application/json" \
  -d '{"prompt": "My name is Alice, email alice@corp.com.", "history": [], "lambda_scale": 0.3}'
```

`/sanitize` and `/sanitize/batch` accept the same optional overrides as `SanitizerConfig`
(`mode`, `surrogate_style`, `lambda_scale`, `min_privacy`, `eta`, `tau`, `sampling_mode`) — they
apply to that request only, without mutating the shared server-side sanitizer.

## Web Dashboard (React/Vite)

```bash
# Terminal 1: API
uvicorn app.main:app --port 8000

# Terminal 2: Web UI
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. CORS is pre-configured for the Vite dev server origin. Set
`VITE_API_URL` if the API runs somewhere other than `http://127.0.0.1:8000`.

The dashboard has two tabs:

- **Text**: paste a prompt, pick *Safe PII redaction* (`pii_only`) or *ProSan+* (`prosan`), and see
  the sanitized prompt with every protected span and its replacement. A Copy button copies the
  result.
- **Image**: drop or choose an image and choose face pixelation, sensitive-text handling and
  optional deep (ProSan+) text analysis. Sensitive text can be replaced with fake values (default),
  boxed, blurred or pixelated. You then view and download the sanitized PNG with its report:
  - faces and text items changed,
  - each original value next to its replacement,
  - PII kinds found and metadata removed,
  - any warnings.

  Metadata is always stripped.

Batch sanitization (`/sanitize/batch`) remains available through the API.

## Edge-case test sets (text and image)

[data/testcases/](data/testcases/README.md) holds 25 labelled text prompts and 25 labelled images.
They are split across three privacy levels: `none` (output must be unchanged), `medium` and
`full`. The edge cases include public names, version numbers that look like IPs, placeholders,
lowercase and accented names, checksummed IDs, secrets in code, tiny and rotated faces, and
metadata-only privacy.

```bash
python benchmark.py --data data/testcases/text_cases.jsonl --show
python run_image_cases.py --show
```

Current results:

| Set | Measured |
|---|---|
| Text (overall) | 93.9% PHR, 79.3% utility, 37.5% benign intact |
| Text, `medium` | 100% PHR |
| Text, `full` | 90.6% PHR |
| Images: faces | 8 of 10 found, 0 false positives (missed: the tiny and the rotated face) |
| Images: metadata | removed in all 5 images that had it |
| Images: text (RapidOCR) | 24 of 34 sensitive strings hidden (medium 4/4, full 20/30); 25 of 25 benign strings still legible |

The low benign figure is a real weakness: public landmarks, fictional characters and tool names
get rewritten. See the test-set README for a per-case breakdown.

## Evaluating on your own prompt dataset

Pass a dataset file to `run_eval.py`; supported formats are JSONL, JSON, CSV, and plain text
(one prompt per line). The default primary dataset is the 1,500-row Microsoft Presidio research
set. It is third-party data and is not committed here (see `.gitignore`); fetch it once:

```bash
git clone --depth 1 https://github.com/microsoft/presidio-research.git /tmp/presidio-research
cp /tmp/presidio-research/data/synth_dataset_v2.json data_presidio.json
python run_eval.py --dataset data_presidio.json --device auto
```

### Results on the Presidio set (1,500 prompts, 2,863 labelled spans, CPU)

| Metric | PII-Only | ProSan+ |
|---|---|---|
| PHR (strict whole-span) | 93.0% | **97.2%** |
| Sensitive-token PHR | 95.7% | **98.3%** |
| Token precision / recall / F1 | 93.2 / 95.6 / 94.4% | 89.9 / 98.0 / 93.8% |
| Collateral (non-sensitive tokens changed) | 2.6% | 4.2% |
| Benign intact (113 PII-free prompts) | 100.0% | **100.0%** |
| PPL ratio (mean, sanitized/original) | 1.322 | 1.557 |
| Latency (s/prompt, warm) | 0.031 | 0.075 |

The set is split by `template_id`: even ids are the development half, odd ids the held-out half
(1,384 spans). One initial failure review covered the whole set; after the split was introduced,
every rule change was driven by development-half misses only, so the held-out half is not fully
blind. ProSan+ PHR is **97.8%** on the development half and **96.5%** on the held-out half — the
figure to quote as generalisation; the 97.2% above is over all 1,500 prompts.

Each JSON/JSONL/CSV row needs prompt text in one of `prompt`, `text`, `full_text`, `original`, or
`input`. Optional `sensitive` and `keep` fields provide gold annotations for privacy and utility
metrics. Presidio-style rows are also supported: `full_text` is the prompt and `spans[].entity_value`
provides the sensitive annotations. TXT files need no labels, so they still produce sanitization,
latency, collateral, and perplexity results, while label-based PHR and utility metrics are omitted.

Example JSONL row:

```json
{"prompt":"Email me at alex@example.com about my appointment.","sensitive":["alex@example.com"],"keep":["appointment"]}
```

The run writes its reports under `eval_results` by default. Choose another directory with
`--out-dir results/my_dataset`.

`run_eval.py` includes token-level accuracy, precision, recall, and F1 against annotated spans.
`benchmark.py` remains a supplemental 45-prompt suite because that set includes `keep` annotations
for measuring task utility.

PHR uses type-aware checks: emails, domains, URLs, and timestamps count as hidden when the entire
identifier is replaced, while other spans require every meaningful token to be edited and absent
from its replacement. The reports retain this strict whole-span PHR and also provide
`sensitive_token_phr_pct`, which counts protected tokens inside partially sanitized spans. Each
annotated span has an audit record with its protected-token count and full-span status, so partial
protection is visible without silently redefining strict PHR. Results also include annotation counts
and exact prompt round-trip restoration. Utility retention is reported only when the dataset supplies
`keep` annotations; it remains N/A for datasets without them.

## Image Sanitization

`sanitizer.image.ImageSanitizer` applies the same privacy goal to images, in three independent
stages that each degrade to a reported warning (never a crash) when an optional dependency is
missing:

* **Faces** — OpenCV's bundled Haar cascade, fully offline; each face is pixelated/blurred/boxed.
* **In-image text** — OCR reads each line with per-character positions. The whole transcript goes
  through the same `find_pii` the text pipeline uses, so cross-line context such as a name under
  "SHIP TO:" counts. Only the characters of each sensitive span are changed.
  - **Default (`replace`):** the span is painted in the local background colour and a realistic fake
    value is drawn in the original ink colour. A screenshot's email becomes a different, plausible
    email and the rest of the line is pixel-identical.
  - **Other styles:** `box`, `blur` and `pixelate`.
  - **OCR engines:** Tesseract is preferred when its binary is on `PATH`. Otherwise RapidOCR, a pip
    package whose models ship in the wheel, runs offline.
  - **Spacing:** RapidOCR often drops spaces ("CallRahulat+9198…"). They are restored from the
    text's structure (camel-case, letter/digit and label joins) before detection.
* **Metadata** — EXIF (GPS, camera, timestamps), ICC profiles and PNG text chunks are dropped.

```bash
python -m app.cli --image photo.jpg scan.png --out-dir sanitized/
python -m app.cli --image id_card.jpg --deep     # also run OCR'd text through the contextual layer
python -m app.cli --image chat.png --text-style box
curl -X POST "http://127.0.0.1:8000/sanitize/image?text_style=replace" -F "file=@photo.jpg" -o sanitized.png
# the report (faces, text replacements, metadata removed, OCR engine, warnings) is in X-Sanitization-Report
```

OpenCV is pinned below 5 because the 5.x wheels no longer ship the Haar cascade files. Use the
single `opencv-python` package: RapidOCR depends on it, and having both it and
`opencv-python-headless` installed lets one overwrite the other's `cv2`. If no OCR engine is
available, the text stage is skipped with a warning and the other two still run.

## Configuration

Settings are fields of `SanitizerConfig` (`sanitizer/config.py`). Pass a config object, or override
fields per call or per API request.

| Key | Meaning |
|---|---|
| `mode` | `prosan` (full pipeline) or `pii_only` (pattern/NER layer only, no LLMs loaded) |
| `surrogate_style` | `realistic` (readable fakes) or `placeholder` (`<PERSON_1>`-style, exactly reversible) |
| `ner_backend` | `transformer` (best recall), `nltk` (offline fallback), `none` |
| `lambda_scale` | λ — caps the *share* of words the contextual layer may touch (a budget, not a quota) |
| `min_privacy` | absolute risk threshold in [0, 1] a word must clear to be considered |
| `max_importance` | protects words above this utility score from replacement |
| `privacy_saturation` | κ — bits of self-information at which surprisal risk saturates |
| `importance_method` | `attention` (1 forward pass, default) or `gradient` (the paper's method, slower) |
| `eta`, `tau`, `top_k`, `sampling_mode` | replacement generation: Eq. 9 weighting, cumulative truncation, candidate pool size, `sample` vs. `max` |
| `device` | `auto` (CUDA when available) / `cpu` / `cuda` |
| `seed` | seeds surrogate generation and stochastic sampling for reproducibility |

Any of these can also be overridden per call without changing the shared instance:

```python
result = sanitizer.sanitize(prompt, config_overrides={"lambda_scale": 0.5, "surrogate_style": "placeholder"})
```

## Multi-turn sessions & restoring an LLM's answer

A `SanitizationSession` keeps surrogates consistent across turns (the same person stays "Alex" in
turn 5 as in turn 1) and remembers what was hidden earlier, so it stays hidden even if a later
turn alone wouldn't flag it:

```python
session = sanitizer.new_session()
r1 = sanitizer.sanitize("My name is Rahul Sharma.", session=session)
r2 = sanitizer.sanitize("Is Rahul eligible for the scheme?", session=session)  # "Rahul" still hidden
```

`r.mapping` (surrogate → original) accumulates across the session and can be used to translate an
LLM's response back into the real values with `PromptSanitizer.restore(text, mapping)`.

## Testing

```bash
pytest -m "not models"   # fast suite, no model downloads
pytest                   # full suite (103 tests; the 4 OCR tests skip only when no OCR engine is installed)
```

## Benchmarking

`data/benchmark.jsonl` is a gold-labelled prompt set (`prompt`, `sensitive` items that must
disappear, `keep` items the task depends on) spanning medical, code, finance, contact and benign
categories.

```bash
python benchmark.py                              # metrics for this checkout
python benchmark.py --baseline /path/to/other/checkout   # compare against another checkout
python benchmark.py --show                       # print every sanitized prompt
```

It reports PHR (sensitive items hidden), task utility, token-level detection accuracy/precision/
recall/F1 against the gold spans, collateral damage, benign-prompt integrity, perplexity ratio,
and per-prompt latency. Token accuracy is shown with precision, recall, and F1 because most prompt
tokens are non-sensitive.

Result vs. the original starter implementation (45 prompts, CPU, `distilgpt2`/`roberta-base`):

| Metric | Baseline | ProSan+ |
|---|---|---|
| PHR (sensitive items hidden) | 86.7% | **96.7%** |
| Utility (task-critical words kept) | 81.0% | **98.0%** |
| Collateral (non-sensitive words changed) | 6.4% | **2.3%** |
| Benign intact (PII-free prompts left untouched) | 25.0% | **91.7%** |
| PPL ratio (sanitized/original fluency) | 1.205 | 1.364 |
| Latency (s/prompt, models warm) | 0.787 | **0.084** |

## What changed vs. the paper / the first implementation

| Problem | Before | Now |
|---|---|---|
| Harmless prompts rewritten | Self-information min-max normalised per prompt, so every prompt has a word at 1.0 | Absolute, calibrated risk; the budget γ_q is a cap, not a quota |
| Rare-but-public words look private ("dyspnea") | Surprisal only | Specificity prior: capitalised non-dictionary tokens ≫ dictionary words |
| First word never selected ("Jack was born…") | No BOS token → first token had 0 bits | BOS prepended |
| Attention importance biased to early tokens | Raw column sums (attention sink) | Received ÷ expected-under-uniform, all layers, BOS excluded |
| Two LM passes for K_w and O_w | 2 forward passes | 1 shared pass |
| MLM copies the name back from another mention / proposes the word itself | Single mask, no filter | All mentions masked together, vocabulary masks, leak filter |
| "John … John" replaced inconsistently | Per-occurrence | Per-entity, per-session (multi-turn) consistency |
| Fake values change between runs | Python's salted `hash()` | Keyed SHA-256 |
| PHR always ≈100% | Computed only over words already removed | Gold-labelled benchmark (`data/benchmark.jsonl`) |
| Card/phone false positives | Regex only | Checksums, context and unit guards |
| Sentence-initial capitalisation treated as identity | Any capitalised word bypassed importance protection | Sentence-initial capitalisation excluded (English orthography, not identity evidence) |
| Multi-word entities half-exposed ("Hawkins, Richardson **and** Santana", "**the corner of** X **and** Y") | One span per name | Firm-name lists, street corners and company suffixes captured as one span |
| Addresses detected in fragments; house number, city and postal lines left visible | Per-fragment spans | House-number prefix, fragment merging and following city/region/postal lines folded into one address |
| Surrogate reused part of the original (`…192` → `192.0.2.x`, `…-22` → `22-…`) | No check | Leak guard: a surrogate may share no identifying token with what it replaces |
| Lower-case and ALL-CAPS names missed by the case-sensitive NER | Single NER pass | Second pass on a length-preserving truecased copy |
| "Wisdom", "Sir", "the Devil" rewritten in benign prompts | Capitalised dictionary word = likely name | Only first names, nationalities or multi-word proper names keep the high prior |

## Known limitations

* Common medical/drug names absent from WordNet (e.g. "paracetamol") can occasionally be treated
  as unusual vocabulary and replaced with a synonym. Closing this fully needs a domain term
  gazetteer rather than a dictionary check.
* `ner_backend: transformer` adds model latency; use `nltk` or `none` for lower-latency or fully
  offline deployments at some cost to recall on names/organizations.
* Nationality adjectives about *things* ("the Russian government", "a French film") are left
  alone by design — only self-descriptions ("I am cambodian", "we are proud saudis") are treated
  as personal data. Presidio labels both, so this accounts for most of the remaining NRP misses.
* Names and addresses in non-Latin scripts (e.g. Greek) are largely missed: the NER model is
  English-only.
