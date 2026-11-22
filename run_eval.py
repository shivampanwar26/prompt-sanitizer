#!/usr/bin/env python3
"""
Real Empirical Evaluation & LaTeX Report Sync Script for ProSan+
Repository: https://github.com/shivampanwar26/prompt-sanitizer

Place this script in the root of your `prompt-sanitizer` repository and run:
    python run_eval.py --device auto

Optional flags:
    --dataset data/benchmark.jsonl     # JSONL, JSON, CSV, or TXT prompt dataset
    --device auto                       # auto uses CUDA when available
    --include-gradient                 # Also benchmark importance_method="gradient"
    --include-placeholder              # Also benchmark surrogate_style="placeholder"
    --images-dir ./sample_images       # Benchmark sanitizer/image_sanitizer.py when added
    --thesis-dir ./ProSan_Plus_LaTeX_Report  # Auto-update Figs/ & Table 4.2 in your thesis
    --out-dir eval_results             # Output directory for JSON, CSV, chart, and .tex table
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is importable
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sanitizer.evaluator import PromptEvaluator
from sanitizer.model_registry import resolve_device
from sanitizer.sanitizer import PromptSanitizer

WORD_RE = re.compile(r"\w+(?:['’.@+\-]\w+)*")

def contains_token(text: str, token: str) -> bool:
    pattern = (r"(?<!\w)" if token[:1].isalnum() else "") + re.escape(token)
    pattern += r"(?!\w)" if token[-1:].isalnum() else ""
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


ADDRESS_STRUCTURE_TOKENS = {
    "street", "st", "avenue", "ave", "boulevard", "blvd", "road", "rd", "drive", "dr",
    "lane", "ln", "way", "place", "pl", "court", "ct", "circle", "cir", "terrace", "ter",
    "trail", "trl", "parkway", "pkwy", "highway", "hwy", "str", "suite", "apt", "apartment",
    "unit", "flat", "house", "plot", "box", "postal", "zip", "code", "address",
}


# Fallback annotated evaluation set used ONLY if no benchmark.jsonl is found on disk
FALLBACK_DATASET: List[Dict[str, Any]] = [
    {
        "category": "healthcare",
        "prompt": "A 30-year-old farmer named James presents with an aortic diameter of 5.0 cm. Genetic testing confirms Marfan Syndrome. What is the best medication?",
        "sensitive": ["James", "30-year-old", "farmer"],
        "keep": ["aortic", "5.0 cm", "Marfan Syndrome", "medication"],
    },
    {
        "category": "healthcare",
        "prompt": "My patient Sarah Connor (email sarah.connor@medmail.org, phone 9876543210) has had a fever of 39.5 C for 2 days. Suggest differential diagnoses.",
        "sensitive": ["Sarah Connor", "sarah.connor@medmail.org", "9876543210"],
        "keep": ["fever", "39.5 C", "differential diagnoses"],
    },
    {
        "category": "code",
        "prompt": "Fix this psycopg2 ImportError: connection = psycopg2.connect(host='114.146.152.14', database='production', user='estrella', password='>j|@X5<pSA')",
        "sensitive": ["114.146.152.14", "estrella", ">j|@X5<pSA"],
        "keep": ["psycopg2", "ImportError", "connect"],
    },
    {
        "category": "code",
        "prompt": "Why does my FastAPI endpoint return 422 when my API key sk-live-9f8e7d6c5b4a3210fedcba9876543210 is passed in the header?",
        "sensitive": ["sk-live-9f8e7d6c5b4a3210fedcba9876543210"],
        "keep": ["FastAPI", "422", "header"],
    },
    {
        "category": "personal",
        "prompt": "My name is Liam Patel and I work as a cardiologist in Seattle. Summarize the key differences between TCP and UDP.",
        "sensitive": ["Liam Patel", "cardiologist", "Seattle"],
        "keep": ["TCP", "UDP", "differences"],
    },
    {
        "category": "corporate",
        "prompt": "Internal memo from Acme Robotics: employee Darius Thompson reported latency spikes on server 10.42.18.99 on 2026-03-14. Draft a root-cause checklist.",
        "sensitive": ["Acme Robotics", "Darius Thompson", "10.42.18.99"],
        "keep": ["latency", "root-cause", "checklist"],
    },
    {
        "category": "benign",
        "prompt": "Explain the time complexity of Dijkstra's shortest path algorithm using a binary heap.",
        "sensitive": [],
        "keep": ["Dijkstra", "shortest path", "binary heap"],
    },
    {
        "category": "benign",
        "prompt": "How does multi-head self-attention scale with sequence length in Transformer models?",
        "sensitive": [],
        "keep": ["multi-head", "self-attention", "Transformer"],
    },
    {
        "category": "benign",
        "prompt": "Write a Python function to compute the eigenvalues of a symmetric matrix using NumPy.",
        "sensitive": [],
        "keep": ["Python", "eigenvalues", "NumPy"],
    },
    {
        "category": "benign",
        "prompt": "What are the differences between L1 and L2 regularization in logistic regression?",
        "sensitive": [],
        "keep": ["L1", "L2", "regularization", "logistic regression"],
    },
]


def _as_list(value: Any) -> List[str]:
    """Normalize optional annotation fields from JSON or CSV values."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith("["):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if item is not None and str(item)]
            except json.JSONDecodeError:
                pass
        return [value]
    return [str(value)]


def _normalize_dataset_row(obj: Dict[str, Any], row_number: int) -> Dict[str, Any]:
    prompt = next((obj.get(key) for key in ("prompt", "text", "full_text", "original", "input")
                   if isinstance(obj.get(key), str) and obj[key].strip()), None)
    if prompt is None:
        raise ValueError(
            f"Dataset row {row_number} has no prompt text. Expected a field named "
            "prompt, text, full_text, original, or input."
        )

    sensitive = _as_list(obj.get("sensitive"))
    sensitive_was_provided = bool(sensitive)
    sensitive_spans: List[Dict[str, Any]] = []
    # Presidio-style records include exact source offsets for every labeled span.
    if isinstance(obj.get("spans"), list):
        for span in obj["spans"]:
            if not isinstance(span, dict):
                continue
            value = span.get("entity_value", span.get("value"))
            start = span.get("start_position", span.get("start"))
            end = span.get("end_position", span.get("end"))
            if value:
                value = str(value)
                if not sensitive_was_provided:
                    sensitive.append(value)
                if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(prompt):
                    if prompt[start:end] == value:
                        sensitive_spans.append({
                            "value": value, "start": start, "end": end,
                            "entity_type": span.get("entity_type", span.get("label", "")),
                        })

    return {
        "category": obj.get("category", obj.get("template_id", "general")),
        "prompt": prompt,
        "sensitive": sensitive,
        "sensitive_spans": sensitive_spans,
        "keep": _as_list(obj.get("keep")),
    }


def load_dataset(dataset_arg: Optional[str]) -> Tuple[List[Dict[str, Any]], str]:
    """Load a prompt dataset from JSONL, JSON, CSV, or one-prompt-per-line TXT."""
    if dataset_arg:
        path = Path(dataset_arg).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            objects = []
            with path.open("r", encoding="utf-8-sig") as f:
                for line_number, line in enumerate(f, 1):
                    if line.strip():
                        obj = json.loads(line)
                        if not isinstance(obj, dict):
                            raise ValueError(f"JSONL row {line_number} must be an object.")
                        objects.append(obj)
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                data = data.get("data", data.get("rows", [data]))
            if not isinstance(data, list):
                raise ValueError("JSON dataset must contain a list of rows, or an object with a 'data' list.")
            objects = data
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                objects = list(csv.DictReader(f))
        elif suffix in {".txt", ".text"}:
            lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            objects = [{"prompt": line} for line in lines]
        else:
            raise ValueError("Unsupported dataset format. Use .jsonl, .json, .csv, or .txt.")

        if any(not isinstance(obj, dict) for obj in objects):
            raise ValueError("Every JSON dataset row must be an object.")
        rows = [_normalize_dataset_row(obj, i) for i, obj in enumerate(objects, 1)]
        if not rows:
            raise ValueError(f"No prompt rows found in dataset: {path}")
        return rows, str(path)

    # The 1,500-row Presidio set is the primary evaluation dataset.
    for path in (REPO_ROOT / "data_presidio.json", REPO_ROOT / "data" / "benchmark.jsonl",
                 REPO_ROOT / "past" / "data" / "benchmark.jsonl"):
        if path.is_file():
            return load_dataset(str(path))
    prompts_txt = REPO_ROOT / "prompts.txt"
    if prompts_txt.is_file():
        return load_dataset(str(prompts_txt))
    return FALLBACK_DATASET, "built-in fallback dataset (10 annotated prompts)"


def find_sensitive_spans(prompt: str, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return gold span offsets, locating string-only annotations when necessary."""
    spans = list(row.get("sensitive_spans", []))
    if spans:
        return spans
    found: List[Dict[str, Any]] = []
    seen = set()
    for value in row.get("sensitive", []):
        if not value:
            continue
        for match in re.finditer(re.escape(value), prompt, flags=re.IGNORECASE):
            key = (match.start(), match.end())
            if key not in seen:
                found.append({
                    "value": prompt[match.start():match.end()], "start": match.start(),
                    "end": match.end(), "entity_type": "",
                })
                seen.add(key)
    return found


def calculate_span_phr(sensitive_spans: List[Dict[str, Any]], edits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score full-span PHR and separately count each protected sensitive token."""
    hidden = 0
    valid = 0
    token_hidden = 0
    token_total = 0
    details = []
    for span in sensitive_spans:
        start, end = int(span["start"]), int(span["end"])
        value = str(span.get("value", ""))
        if not value or start < 0 or end <= start:
            continue
        entity_type = str(span.get("entity_type", "")).upper()
        all_tokens = list(re.finditer(r"\w+", value, flags=re.UNICODE))
        tokens = [match for match in all_tokens
                  if len(match.group()) > 2 or any(char.isdigit() for char in match.group())]
        if entity_type == "STREET_ADDRESS":
            tokens = [match for match in tokens if match.group().casefold() not in ADDRESS_STRUCTURE_TOKENS]
        # Keep short standalone identifiers (for example, a two-letter state code)
        # scoreable; skip short connector words only inside longer annotated spans.
        if not tokens and entity_type != "STREET_ADDRESS":
            tokens = all_tokens
        if not tokens:
            continue
        valid += 1
        token_total += len(tokens)
        hidden_for_span = 0
        if entity_type in {"EMAIL_ADDRESS", "DOMAIN_NAME", "URL", "DATE_TIME"}:
            # Shared URL/email syntax and time components are not independent
            # identifiers; these annotated values must be replaced atomically.
            edit = next((e for e in edits
                         if int(e.get("start", -1)) <= start
                         and int(e.get("end", -1)) >= end), None)
            if edit is not None and value.casefold() not in str(edit.get("replacement", "")).casefold():
                hidden += 1
                hidden_for_span = len(tokens)
                token_hidden += len(tokens)
            details.append({
                "start": start, "end": end, "entity_type": entity_type, "value": value,
                "tokens_hidden": hidden_for_span, "tokens_total": len(tokens),
                "fully_hidden": hidden_for_span == len(tokens),
            })
            continue
        all_tokens_changed = True
        for token in tokens:
            token_start, token_end = start + token.start(), start + token.end()
            edit = next((e for e in edits
                         if int(e.get("start", -1)) <= token_start
                         and int(e.get("end", -1)) >= token_end), None)
            if edit is None or contains_token(str(edit.get("replacement", "")), token.group()):
                all_tokens_changed = False
            else:
                hidden_for_span += 1
        token_hidden += hidden_for_span
        if all_tokens_changed:
            hidden += 1
        details.append({
            "start": start, "end": end, "entity_type": entity_type, "value": value,
            "tokens_hidden": hidden_for_span, "tokens_total": len(tokens),
            "fully_hidden": all_tokens_changed,
        })
    return {
        "total": valid,
        "hidden": hidden,
        "phr": round(100.0 * hidden / valid, 2) if valid else 0.0,
        "token_total": token_total,
        "token_hidden": token_hidden,
        "token_phr": round(100.0 * token_hidden / token_total, 2) if token_total else 0.0,
        "details": details,
    }


def collateral_counts(original: str, sanitized: str, sensitive_items: List[str]) -> Tuple[int, int]:
    """Count original non-sensitive tokens absent from the sanitized prompt."""
    orig_tokens = WORD_RE.findall(original)
    if not orig_tokens:
        return 0, 0

    sens_tokens = set()
    for item in sensitive_items:
        for tok in WORD_RE.findall(item):
            sens_tokens.add(tok.lower())

    non_sens = [t for t in orig_tokens if t.lower() not in sens_tokens]
    if not non_sens:
        return 0, 0

    san_lower = {t.lower() for t in WORD_RE.findall(sanitized)}
    changed = sum(1 for t in non_sens if t.lower() not in san_lower)
    return changed, len(non_sens)


def token_detection_counts(
    prompt: str, sensitive_spans: List[Dict[str, Any]], edits: List[Dict[str, Any]],
) -> Tuple[int, int, int, int]:
    """Token-level TP, FP, FN, TN based on overlap with gold and edited spans."""
    gold_ranges = [(int(s["start"]), int(s["end"])) for s in sensitive_spans]
    predicted_ranges = [(int(e["start"]), int(e["end"])) for e in edits]
    tp = fp = fn = tn = 0
    for token in WORD_RE.finditer(prompt):
        gold = any(token.start() < end and token.end() > start for start, end in gold_ranges)
        predicted = any(token.start() < end and token.end() > start for start, end in predicted_ranges)
        if gold and predicted:
            tp += 1
        elif predicted:
            fp += 1
        elif gold:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def build_sanitizer(
    mode: str,
    device: str,
    importance_method: str = "attention",
    surrogate_style: str = "realistic",
) -> PromptSanitizer:
    """
    Instantiate the sanitizer on the requested device before model loading.
    """
    sanitizer = PromptSanitizer(device=device)
    sanitizer.config.mode = mode
    sanitizer.config.device = device
    sanitizer.config.importance_method = importance_method
    sanitizer.config.surrogate_style = surrogate_style

    # Reload causal analyzer if device or importance_method changed
    if mode == "prosan":
        sanitizer._load_prosan_components()
    return sanitizer


def evaluate_config(
    name: str,
    rows: List[Dict[str, Any]],
    evaluator: PromptEvaluator,
    mode: str,
    device: str,
    importance_method: str = "attention",
    surrogate_style: str = "realistic",
) -> Dict[str, Any]:
    """Run real end-to-end evaluation for a single ProSan+ configuration."""
    print(f"\n[Evaluating] {name} (mode={mode}, importance={importance_method}, style={surrogate_style}, device={device})")
    sanitizer = build_sanitizer(
        mode=mode,
        device=device,
        importance_method=importance_method,
        surrogate_style=surrogate_style,
    )

    # Warm up models before timing
    sanitizer.sanitize("Warm up the models, John.", evaluate=False)

    per_prompt: List[Dict[str, Any]] = []
    total_sens = 0
    total_hidden = 0
    total_sensitive_tokens = 0
    total_sensitive_tokens_hidden = 0
    unmatched_sensitive_labels = 0
    total_keep = 0
    total_kept = 0
    benign_total = 0
    benign_intact = 0
    collateral_changed_tokens = 0
    collateral_total_tokens = 0
    ppl_ratios: List[float] = []
    latencies: List[float] = []
    restore_total = 0
    restore_exact = 0
    token_tp = token_fp = token_fn = token_tn = 0

    for idx, row in enumerate(rows):
        prompt = row["prompt"]
        sensitive = row.get("sensitive", [])
        keep = row.get("keep", [])

        t0 = time.perf_counter()
        result = sanitizer.sanitize(prompt, evaluate=False)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        sanitized = result.text

        # PHR
        sensitive_spans = find_sensitive_spans(prompt, row)
        unmatched_sensitive_labels += max(0, len(sensitive) - len(sensitive_spans))
        phr_res = calculate_span_phr(sensitive_spans, result.selected_words) if sensitive_spans else None
        tp, fp, fn, tn = token_detection_counts(prompt, sensitive_spans, result.selected_words)
        token_tp += tp
        token_fp += fp
        token_fn += fn
        token_tn += tn
        if phr_res:
            total_sens += int(phr_res["total"])
            total_hidden += int(phr_res["hidden"])
            total_sensitive_tokens += int(phr_res["token_total"])
            total_sensitive_tokens_hidden += int(phr_res["token_hidden"])

        # Utility retention
        util_res = PromptEvaluator.utility_retention(keep, sanitized) if keep else None
        if util_res:
            total_keep += int(util_res["total"])
            total_kept += int(util_res["kept"])

        # Benign intact
        if not sensitive:
            benign_total += 1
            if sanitized == prompt:
                benign_intact += 1

        # Collateral rate
        collateral_changed, collateral_total = collateral_counts(prompt, sanitized, sensitive)
        collat = collateral_changed / collateral_total if collateral_total else 0.0
        collateral_changed_tokens += collateral_changed
        collateral_total_tokens += collateral_total

        # Perplexity ratio
        orig_ppl, san_ppl = evaluator.perplexity_batch([prompt, sanitized])
        ppl_ratio = (san_ppl / orig_ppl) if orig_ppl and orig_ppl > 0 else 1.0
        ppl_ratios.append(ppl_ratio)

        # Bidirectional restoration check
        restored = PromptSanitizer.restore(sanitized, result.mapping) if result.mapping else sanitized
        if result.mapping:
            restore_total += 1
            if restored == prompt:
                restore_exact += 1

        per_prompt.append({
            "index": idx,
            "category": row.get("category", "general"),
            "original": prompt,
            "sanitized": sanitized,
            "restored": restored,
            "selected_words": [e.get("word") for e in result.selected_words],
            "selected_edits": result.selected_words,
            "mapping": result.mapping,
            "H_q": result.H_q,
            "gamma_q": result.gamma_q,
            "phr": phr_res["phr"] if phr_res else None,
            "sensitive_span_count": phr_res["total"] if phr_res else 0,
            "sensitive_spans_hidden": phr_res["hidden"] if phr_res else 0,
            "sensitive_token_total": phr_res["token_total"] if phr_res else 0,
            "sensitive_tokens_hidden": phr_res["token_hidden"] if phr_res else 0,
            "sensitive_token_phr_pct": phr_res["token_phr"] if phr_res else None,
            "sensitive_span_audit": phr_res["details"] if phr_res else [],
            "unmatched_sensitive_labels": max(0, len(sensitive) - len(sensitive_spans)),
            "utility": util_res["retention"] if util_res else None,
            "collateral": round(collat, 4),
            "orig_ppl": orig_ppl,
            "san_ppl": san_ppl,
            "ppl_ratio": round(ppl_ratio, 4),
            "latency_s": round(elapsed, 4),
        })

    token_total = token_tp + token_fp + token_fn + token_tn
    token_precision = token_tp / (token_tp + token_fp) if token_tp + token_fp else 0.0
    token_recall = token_tp / (token_tp + token_fn) if token_tp + token_fn else 0.0
    token_f1 = 2 * token_precision * token_recall / (token_precision + token_recall) if token_precision + token_recall else 0.0

    summary = {
        "name": name,
        "mode": mode,
        "importance_method": importance_method,
        "surrogate_style": surrogate_style,
        "num_prompts": len(rows),
        "num_prompts_with_sensitive_spans": sum(bool(find_sensitive_spans(r["prompt"], r)) for r in rows),
        "num_sensitive_spans": total_sens,
        "num_sensitive_labels_unmatched": unmatched_sensitive_labels,
        "num_prompts_without_sensitive_labels": benign_total,
        "num_prompts_with_utility_labels": sum(bool(r.get("keep")) for r in rows),
        "phr_pct": round(100.0 * total_hidden / total_sens, 1) if total_sens else None,
        "sensitive_token_phr_pct": round(100.0 * total_sensitive_tokens_hidden / total_sensitive_tokens, 1) if total_sensitive_tokens else None,
        "token_accuracy_pct": round(100.0 * (token_tp + token_tn) / token_total, 1) if token_total else None,
        "token_precision_pct": round(100.0 * token_precision, 1),
        "token_recall_pct": round(100.0 * token_recall, 1),
        "token_f1_pct": round(100.0 * token_f1, 1),
        "utility_pct": round(100.0 * total_kept / total_keep, 1) if total_keep else None,
        "collateral_pct": round(100.0 * collateral_changed_tokens / collateral_total_tokens, 1) if collateral_total_tokens else 0.0,
        "benign_intact_pct": round(100.0 * benign_intact / benign_total, 1) if benign_total else None,
        "ppl_ratio": round(statistics.mean(ppl_ratios), 3) if ppl_ratios else 1.0,
        "latency_mean_s": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "latency_median_s": round(statistics.median(latencies), 3) if latencies else 0.0,
        "restore_pct": round(100.0 * restore_exact / restore_total, 1) if restore_total else None,
    }
    return {"summary": summary, "per_prompt": per_prompt}


def evaluate_images_if_available(images_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    """Optional hook: benchmarks sanitizer/image_sanitizer.py once implemented."""
    if not images_dir:
        return None
    img_path = Path(images_dir)
    if not img_path.is_dir():
        print(f"[Images] Directory not found: {img_path} (skipping image benchmark)")
        return None
    try:
        from sanitizer.image_sanitizer import ImageSanitizer  # type: ignore
    except ImportError:
        print("[Images] sanitizer/image_sanitizer.py not yet implemented; skipping image benchmark.")
        return None

    image_files = [
        p for p in sorted(img_path.iterdir())
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    if not image_files:
        print(f"[Images] No images found in {img_path}")
        return None

    img_san = ImageSanitizer()
    latencies: List[float] = []
    exif_stripped = 0
    for fpath in image_files:
        t0 = time.perf_counter()
        out = img_san.sanitize_image(str(fpath))
        latencies.append(time.perf_counter() - t0)
        if getattr(out, "exif_stripped", True):
            exif_stripped += 1

    return {
        "num_images": len(image_files),
        "exif_stripped_pct": round(100.0 * exif_stripped / len(image_files), 1),
        "latency_mean_s": round(statistics.mean(latencies), 3),
    }


def generate_latex_table(baseline: Dict[str, Any], prosan: Dict[str, Any], num_prompts: int) -> str:
    """Build Table 4.2 LaTeX code directly from real measured summary metrics."""
    def fmt_pct(val: Optional[float]) -> str:
        return f"{val:.1f}\\%" if val is not None else "--"

    def fmt_diff_pp(b: Optional[float], p: Optional[float]) -> str:
        if b is None or p is None:
            return "--"
        d = p - b
        sign = "+" if d >= 0 else "-"
        return f"${sign}{abs(d):.1f}$ pp"

    b_lat = baseline["latency_mean_s"]
    p_lat = prosan["latency_mean_s"]
    speedup = f"{b_lat / p_lat:.1f}$\\times$" if p_lat > 0 else "--"
    ppl_diff = prosan["ppl_ratio"] - baseline["ppl_ratio"]
    ppl_sign = "+" if ppl_diff >= 0 else "-"

    return rf"""\begin{{table}}[!htbp]
\centering
\small
\setlength{{\tabcolsep}}{{5pt}}
\renewcommand{{\arraystretch}}{{1.18}}
\caption{{Empirical comparison of {baseline["name"]} vs.\ {prosan["name"]} measured on {num_prompts} prompts (\texttt{{run\_eval.py}}).}}
\label{{tab:repo_benchmark}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}
  >{{\raggedright\arraybackslash}}p{{0.18\linewidth}}
  >{{\raggedright\arraybackslash}}X
  >{{\centering\arraybackslash}}p{{0.11\linewidth}}
  >{{\centering\arraybackslash}}p{{0.11\linewidth}}
  >{{\centering\arraybackslash}}p{{0.14\linewidth}}
@{{}}}}
\toprule
\textbf{{Metric}} & \textbf{{Interpretation}} & \textbf{{{baseline["name"]}}} & \textbf{{{prosan["name"]}}} & \textbf{{Difference}} \\
\midrule
\textbf{{PHR (\%)}} & Sensitive items hidden (higher is better) & {fmt_pct(baseline["phr_pct"])} & \textbf{{{fmt_pct(prosan["phr_pct"])}}} & {fmt_diff_pp(baseline["phr_pct"], prosan["phr_pct"])} \\
\textbf{{Sensitive-token PHR (\%)}} & Sensitive tokens changed, including partial-span protection & {fmt_pct(baseline["sensitive_token_phr_pct"])} & \textbf{{{fmt_pct(prosan["sensitive_token_phr_pct"])}}} & {fmt_diff_pp(baseline["sensitive_token_phr_pct"], prosan["sensitive_token_phr_pct"])} \\
\textbf{{Token accuracy (\%)}} & Token-level detection accuracy & {fmt_pct(baseline["token_accuracy_pct"])} & \textbf{{{fmt_pct(prosan["token_accuracy_pct"])}}} & {fmt_diff_pp(baseline["token_accuracy_pct"], prosan["token_accuracy_pct"])} \\
\textbf{{Token F1 (\%)}} & Token-level precision/recall balance & {fmt_pct(baseline["token_f1_pct"])} & \textbf{{{fmt_pct(prosan["token_f1_pct"])}}} & {fmt_diff_pp(baseline["token_f1_pct"], prosan["token_f1_pct"])} \\
\textbf{{Utility (\%)}} & Task-critical words kept (higher is better) & {fmt_pct(baseline["utility_pct"])} & \textbf{{{fmt_pct(prosan["utility_pct"])}}} & {fmt_diff_pp(baseline["utility_pct"], prosan["utility_pct"])} \\
\textbf{{Collateral (\%)}} & Non-sensitive words changed (lower is better) & {fmt_pct(baseline["collateral_pct"])} & \textbf{{{fmt_pct(prosan["collateral_pct"])}}} & {fmt_diff_pp(baseline["collateral_pct"], prosan["collateral_pct"])} \\
\textbf{{Benign Intact (\%)}} & PII-free prompts untouched (higher is better) & {fmt_pct(baseline["benign_intact_pct"])} & \textbf{{{fmt_pct(prosan["benign_intact_pct"])}}} & {fmt_diff_pp(baseline["benign_intact_pct"], prosan["benign_intact_pct"])} \\
\textbf{{PPL Ratio}} & Sanitized/original perplexity (closer to 1.0) & {baseline["ppl_ratio"]:.3f} & {prosan["ppl_ratio"]:.3f} & ${ppl_sign}{abs(ppl_diff):.3f}$ \\
\textbf{{Warm Latency}} & Seconds per prompt (lower is better) & {b_lat:.3f}~s & \textbf{{{p_lat:.3f}~s}} & {speedup} \\
\bottomrule
\end{{tabularx}}
\end{{table}}"""


def generate_benchmark_chart(baseline: Dict[str, Any], prosan: Dict[str, Any], out_dir: Path) -> Optional[Path]:
    """Generate benchmark_comparison.pdf/.png from real measured results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[Chart] matplotlib not installed; skipping chart generation.")
        return None

    metrics = [
        ("PHR\n(Higher better)", baseline["phr_pct"] or 0.0, prosan["phr_pct"] or 0.0),
        ("Utility\nRetention\n(Higher better)", baseline["utility_pct"] or 0.0, prosan["utility_pct"] or 0.0),
        ("Benign\nIntact\n(Higher better)", baseline["benign_intact_pct"] or 0.0, prosan["benign_intact_pct"] or 0.0),
        ("Collateral\nChange\n(Lower better)", baseline["collateral_pct"] or 0.0, prosan["collateral_pct"] or 0.0),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=300, gridspec_kw={"width_ratios": [2.2, 1.1]})
    x = np.arange(len(metrics))
    w = 0.35

    b_vals = [m[1] for m in metrics]
    p_vals = [m[2] for m in metrics]

    bars1 = ax1.bar(x - w / 2, b_vals, w, label=baseline["name"], color="#94A3B8", edgecolor="#475569", linewidth=0.7)
    bars2 = ax1.bar(x + w / 2, p_vals, w, label=prosan["name"], color="#2563EB", edgecolor="#1E40AF", linewidth=0.7)

    ax1.set_ylabel("Percentage (%)", fontweight="bold")
    ax1.set_title(f"(a) Measured Privacy, Utility & Collateral ({prosan['num_prompts']} Prompts)", fontweight="bold", fontsize=10, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[0] for m in metrics], fontsize=8.5)
    ax1.set_ylim(0, 114)
    ax1.grid(axis="y", linestyle="--", alpha=0.45)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=True, facecolor="white", loc="upper right", fontsize=8.5)

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    lat_labels = [baseline["name"], prosan["name"]]
    lat_vals = [baseline["latency_mean_s"], prosan["latency_mean_s"]]
    bars_l = ax2.bar(lat_labels, lat_vals, width=0.48, color=["#94A3B8", "#0D9488"], edgecolor="#334155", linewidth=0.8)
    ax2.set_ylabel("Latency (seconds / prompt)", fontweight="bold")
    ax2.set_title("(b) Warm Inference Latency", fontweight="bold", fontsize=10, pad=10)
    ax2.set_ylim(0, max(lat_vals) * 1.28 + 0.05)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    ax2.set_axisbelow(True)

    for bar, val in zip(bars_l, lat_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + max(lat_vals) * 0.03, f"{val:.3f} s", ha="center", va="bottom", fontsize=8.8, fontweight="bold")

    ax2.annotate(
        f"PPL Ratio:\n{baseline['name']}: {baseline['ppl_ratio']:.3f}\n{prosan['name']}: {prosan['ppl_ratio']:.3f}",
        xy=(0.98, 0.92),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )

    plt.tight_layout(pad=1.0)
    pdf_path = out_dir / "benchmark_comparison.pdf"
    png_path = out_dir / "benchmark_comparison.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def sync_to_thesis(thesis_dir: Path, tex_table: str, chart_pdf: Optional[Path]) -> None:
    """Optionally update Figs/benchmark_comparison.pdf and Table 4.2 in Chapter4/chapter4.tex."""
    if chart_pdf and (thesis_dir / "Figs").is_dir():
        shutil.copy2(chart_pdf, thesis_dir / "Figs" / "benchmark_comparison.pdf")
        png_src = chart_pdf.with_suffix(".png")
        if png_src.is_file():
            shutil.copy2(png_src, thesis_dir / "Figs" / "benchmark_comparison.png")
        print(f"[Thesis Sync] Updated {thesis_dir / 'Figs' / 'benchmark_comparison.pdf'}")

    ch4_path = thesis_dir / "Chapter4" / "chapter4.tex"
    if ch4_path.is_file():
        ch4 = ch4_path.read_text(encoding="utf-8")
        pattern = re.compile(
            r"\\begin\{table\}\[!htbp\].*?\\label\{tab:repo_benchmark\}.*?\\end\{table\}",
            flags=re.DOTALL,
        )
        if pattern.search(ch4):
            updated = pattern.sub(lambda _: tex_table, ch4, count=1)
            ch4_path.write_text(updated, encoding="utf-8")
            print(f"[Thesis Sync] Replaced tab:repo_benchmark in {ch4_path}")
        else:
            print(f"[Thesis Sync] Could not find \\label{{tab:repo_benchmark}} in {ch4_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real empirical evaluation on prompt-sanitizer and export thesis artifacts.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset file (.jsonl, .json, .csv, or .txt); rows need prompt/text/full_text/original/input.")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"], help="Inference device (auto uses CUDA when available).")
    parser.add_argument("--include-gradient", action="store_true", help="Also evaluate importance_method='gradient'.")
    parser.add_argument("--include-placeholder", action="store_true", help="Also evaluate surrogate_style='placeholder'.")
    parser.add_argument("--images-dir", type=str, default=None, help="Optional directory of images for image_sanitizer benchmark.")
    parser.add_argument("--thesis-dir", type=str, default=None, help="Optional path to unpacked ProSan_Plus_LaTeX_Report folder.")
    parser.add_argument("--out-dir", type=str, default="eval_results", help="Directory for output files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, source_label = load_dataset(args.dataset)
    print(f"Loaded {len(rows)} prompts from: {source_label}")

    device = str(resolve_device(args.device))
    print(f"Using inference device: {device}")

    evaluator = PromptEvaluator(model_name="distilgpt2", device=device)

    configs: List[Tuple[str, str, str, str]] = [
        ("PII-Only", "pii_only", "attention", "realistic"),
        ("ProSan+", "prosan", "attention", "realistic"),
    ]
    if args.include_gradient:
        configs.append(("ProSan+ (Grad)", "prosan", "gradient", "realistic"))
    if args.include_placeholder:
        configs.append(("ProSan+ (Placeholder)", "prosan", "attention", "placeholder"))

    all_results: Dict[str, Any] = {
        "dataset_source": source_label,
        "device": device,
        "runs": {},
    }

    summaries: List[Dict[str, Any]] = []
    for name, mode, imp, style in configs:
        res = evaluate_config(
            name=name,
            rows=rows,
            evaluator=evaluator,
            mode=mode,
            device=device,
            importance_method=imp,
            surrogate_style=style,
        )
        all_results["runs"][name] = res
        summaries.append(res["summary"])

    img_summary = evaluate_images_if_available(args.images_dir)
    if img_summary:
        all_results["image_benchmark"] = img_summary

    # Print clean console summary
    print("\n================ REAL EVALUATION SUMMARY ================")
    header = f"{'Config':<22} {'PHR%':>7} {'Acc%':>7} {'F1%':>7} {'Util%':>7} {'Collat%':>8} {'NoLabel%':>9} {'PPL':>7} {'Lat(s)':>8} {'Rest%':>7}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        fmt = lambda v: f"{v:.1f}" if v is not None else "N/A"
        print(
            f"{s['name']:<22} {fmt(s['phr_pct']):>7} {fmt(s['token_accuracy_pct']):>7} {fmt(s['token_f1_pct']):>7} "
            f"{fmt(s['utility_pct']):>7} {fmt(s['collateral_pct']):>8} {fmt(s['benign_intact_pct']):>9} "
            f"{s['ppl_ratio']:>7.3f} {s['latency_mean_s']:>8.3f} {fmt(s['restore_pct']):>7}"
        )

    # Save JSON
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # Save CSV
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    # Generate real LaTeX Table 4.2 & chart comparing PII-Only vs ProSan+
    tex_table = generate_latex_table(summaries[0], summaries[1], len(rows))
    tex_path = out_dir / "table_4_2_real.tex"
    tex_path.write_text(tex_table, encoding="utf-8")

    chart_pdf = generate_benchmark_chart(summaries[0], summaries[1], out_dir)

    print(f"\nSaved real outputs to: {out_dir.resolve()}")
    print(f"  - {json_path.name}")
    print(f"  - {csv_path.name}")
    print(f"  - {tex_path.name}")
    if chart_pdf:
        print(f"  - {chart_pdf.name} (+ .png)")

    if args.thesis_dir:
        sync_to_thesis(Path(args.thesis_dir), tex_table, chart_pdf)


if __name__ == "__main__":
    main()
