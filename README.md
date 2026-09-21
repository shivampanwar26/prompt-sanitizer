# ProSan — Utility-Based Prompt Privacy Sanitizer

A clean, unified implementation of the ProSan prompt-sanitization pipeline described in *"ProSan: Utility-Based Prompt Privacy Sanitizer"* (IEEE TIFS 2026).

## Pipeline Workflow

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

## What changed vs. the paper / the first implementation (ProSan+)

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

## Benchmark

```bash
python benchmark.py                    # this checkout
python benchmark.py --baseline PATH    # compare with another checkout (e.g. the original commit)
python benchmark.py --show             # print every sanitized prompt
```

## Modes and styles

* `mode: prosan` (default) runs the full pipeline. `mode: pii_only` never loads the language models.
* `surrogate_style: realistic` (default) produces readable fakes. `placeholder` produces `<PERSON_1>`-style tags that restore exactly.

```python
from sanitizer import PromptSanitizer

s = PromptSanitizer()
session = s.new_session()                       # keeps surrogates consistent across turns
r = s.sanitize("My name is Rahul Sharma, call +91 98765 43210", session=session)
answer = call_your_llm(r.text)
print(s.restore(answer, r.mapping))             # real names/values back in the answer
```

## Quick Start & Requirements

Python 3.10+ recommended.

```bash
# Setup Virtual Environment
python -m venv .venv

# Activate:
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Default models loaded on first run:
- Importance & Privacy Self-Information: `distilgpt2`
- Candidate Replacement Generator: `roberta-base`
- Named-entity recognition: `dslim/distilbert-NER` (set `ner_backend: nltk` or `none` to skip it)

Tests: `pytest -m "not models"` runs the fast suite without downloading models. Plain `pytest` runs everything.

---

## Running the CLI

Run direct prompt desensitization via the clean CLI:

```bash
python -m app.cli "My name is John Doe and my email is john@example.com. I live in Delhi and have severe chest pain."
```

Or pass a file containing prompts (one per line):

```bash
python -m app.cli -f prompts.txt
```

Or evaluate readability perplexity ($PPL$):

```bash
python -m app.cli --eval "My name is John Doe and I live in Delhi."
```

---

## Running the REST API

```bash
uvicorn app.main:app --reload
```

Send a POST request to `/sanitize`:

```json
{
  "prompt": "My name is John Doe and my email is john@example.com."
}
```

---

## Web Dashboard (React/Vite)

Start the API and frontend in separate terminals:

```bash
# Terminal 1: API
uvicorn app.main:app --port 8000

# Terminal 2: Web UI
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your web browser.
