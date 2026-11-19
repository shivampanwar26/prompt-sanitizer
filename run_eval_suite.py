"""Automated Benchmark & Test Suite for ProSan.

Uses benchmark datasets:
1. Microsoft Presidio Research Dataset (data_presidio.json) - 1,500 ground-truth PII prompts.
2. OpenAI HumanEval Dataset (data_humaneval.jsonl.gz) - 164 functional programming tasks.

Calculates:
- Privacy Hiding Rate (PHR) / Entity Recall against ground-truth labels.
- Perplexity (PPL) ratio (Readability / Utility preservation).
- GPU Latency per prompt.
"""

import gzip
import json
import os
import sys
import time
from typing import Dict, List, Any

# Ensure offline transformers so it doesn't wait on HF network checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sanitizer import PromptSanitizer


def load_presidio_dataset(limit: int = 15) -> List[Dict[str, Any]]:
    """Load official Microsoft Presidio PII benchmark dataset from local file."""
    path = "data_presidio.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"[+] Loaded {len(raw_data)} ground-truth labeled samples from Microsoft Presidio.")

    target_entities = {
        "PERSON", "STREET_ADDRESS", "PHONE_NUMBER",
        "EMAIL_ADDRESS", "DATE_TIME", "IP_ADDRESS", "CREDIT_CARD"
    }
    filtered = []
    for item in raw_data:
        entities = [s["entity_type"] for s in item.get("spans", [])]
        if any(e in target_entities for e in entities) and len(item["full_text"].strip()) > 15:
            filtered.append(item)
            if len(filtered) >= limit:
                break

    print(f"[+] Selected {len(filtered)} targeted PII test samples for evaluation.")
    return filtered


def load_humaneval_dataset(limit: int = 10) -> List[Dict[str, Any]]:
    """Load OpenAI HumanEval dataset for coding utility preservation testing."""
    path = "data_humaneval.jsonl.gz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")

    with gzip.open(path, "rt", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    print(f"[+] Loaded {len(lines)} programming tasks from OpenAI HumanEval.")
    selected = lines[:limit]
    print(f"[+] Selected {len(selected)} code samples for utility testing.")
    return selected


def run_presidio_benchmark(sanitizer: PromptSanitizer, samples: List[Dict[str, Any]]) -> Dict[str, float]:
    print("\n" + "=" * 80)
    print(" SUITE 1: MICROSOFT PRESIDIO PII GROUND-TRUTH BENCHMARK")
    print("=" * 80)

    total_gt_entities = 0
    total_hidden_entities = 0
    latencies = []
    ppl_ratios = []

    for idx, sample in enumerate(samples, 1):
        prompt = sample["full_text"]
        gt_spans = sample.get("spans", [])

        # Ground truth entity values to verify
        gt_values = [s["entity_value"].strip() for s in gt_spans if len(s["entity_value"].strip()) > 1]
        if not gt_values:
            continue

        t0 = time.time()
        result = sanitizer.sanitize(prompt, evaluate=True)
        dur = time.time() - t0
        latencies.append(dur)

        hidden = 0
        for val in gt_values:
            if val.lower() not in result.text.lower():
                hidden += 1

        total_gt_entities += len(gt_values)
        total_hidden_entities += hidden
        sample_phr = (hidden / len(gt_values)) * 100.0 if gt_values else 100.0

        if result.original_perplexity and result.original_perplexity > 0 and result.perplexity:
            ratio = result.perplexity / result.original_perplexity
            ppl_ratios.append(ratio)

        clean_in = prompt.replace("\n", " ")[:80]
        clean_out = result.text.replace("\n", " ")[:80]
        print(f"\n[Test #{idx:02d}] Latency: {dur:.3f}s | Ground-Truth Hidden: {hidden}/{len(gt_values)} ({sample_phr:.1f}%)")
        print(f"  Input:     {clean_in}...")
        print(f"  Sanitized: {clean_out}...")
        if result.selected_words:
            repls = [f"'{w['word']}'->'{w.get('replacement')}'" for w in result.selected_words[:3]]
            print(f"  Changes:   {', '.join(repls)}" + ("..." if len(result.selected_words) > 3 else ""))

    overall_phr = (total_hidden_entities / total_gt_entities) * 100.0 if total_gt_entities else 100.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_ppl_ratio = sum(ppl_ratios) / len(ppl_ratios) if ppl_ratios else 1.0

    print("\n" + "-" * 80)
    print(" Presidio PII Benchmark Results:")
    print(f"   Total Ground-Truth Entities Tested : {total_gt_entities}")
    print(f"   Entities Successfully Redacted     : {total_hidden_entities}")
    print(f"   Ground-Truth Entity Recall (PHR)   : {overall_phr:.2f}%")
    print(f"   Average Latency per Prompt         : {avg_latency:.3f} seconds")
    print(f"   Average PPL Ratio (Sanitized/Orig) : {avg_ppl_ratio:.4f}")
    print("-" * 80)

    return {
        "phr": overall_phr,
        "latency": avg_latency,
        "ppl_ratio": avg_ppl_ratio,
        "total_entities": total_gt_entities,
    }


def run_code_utility_benchmark(sanitizer: PromptSanitizer, samples: List[Dict[str, Any]]) -> Dict[str, float]:
    print("\n" + "=" * 80)
    print(" SUITE 2: OPENAI HUMANEVAL CODE UTILITY PRESERVATION BENCHMARK")
    print("=" * 80)

    latencies = []
    ppl_ratios = []
    corrupted_functions = 0

    for idx, sample in enumerate(samples, 1):
        code_prompt = sample["prompt"]
        entry_point = sample.get("entry_point", "")

        t0 = time.time()
        result = sanitizer.sanitize(code_prompt, evaluate=True)
        dur = time.time() - t0
        latencies.append(dur)

        entry_preserved = entry_point in result.text if entry_point else True
        if not entry_preserved:
            corrupted_functions += 1

        if result.original_perplexity and result.original_perplexity > 0 and result.perplexity:
            ratio = result.perplexity / result.original_perplexity
            ppl_ratios.append(ratio)

        first_line = code_prompt.strip().splitlines()[0] if code_prompt.strip().splitlines() else ""
        print(f"\n[Code #{idx:02d}] Task: {sample.get('task_id', '')} | Latency: {dur:.3f}s | Signature Preserved: {entry_preserved}")
        print(f"  First Line: {first_line}")
        if result.selected_words:
            print(f"  Replacements applied: {len(result.selected_words)}")
        else:
            print(f"  Replacements applied: 0 (Preserved intact)")

    integrity_rate = ((len(samples) - corrupted_functions) / len(samples)) * 100.0 if samples else 100.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_ppl_ratio = sum(ppl_ratios) / len(ppl_ratios) if ppl_ratios else 1.0

    print("\n" + "-" * 80)
    print(" HumanEval Code Utility Benchmark Results:")
    print(f"   Total Code Tasks Tested            : {len(samples)}")
    print(f"   Signature Integrity Rate           : {integrity_rate:.2f}%")
    print(f"   Average Latency per Prompt         : {avg_latency:.3f} seconds")
    print(f"   Average PPL Ratio (Sanitized/Orig) : {avg_ppl_ratio:.4f}")
    print("-" * 80)

    return {
        "integrity_rate": integrity_rate,
        "latency": avg_latency,
        "ppl_ratio": avg_ppl_ratio,
    }


def main():
    print("================================================================================")
    print(" ProSan Automated Benchmark & Test Suite")
    print("================================================================================")

    presidio_samples = load_presidio_dataset(limit=15)
    humaneval_samples = load_humaneval_dataset(limit=10)

    print("\n[*] Initializing PromptSanitizer on CUDA...")
    sanitizer = PromptSanitizer()

    # Warm-up
    sanitizer.sanitize("warm up test", evaluate=True)
    print("[+] Model loaded and warmed up.")

    presidio_results = run_presidio_benchmark(sanitizer, presidio_samples)
    code_results = run_code_utility_benchmark(sanitizer, humaneval_samples)

    print("\n" + "=" * 80)
    print(" OVERALL BENCHMARK SUITE SUMMARY")
    print("=" * 80)
    print(f" Microsoft Presidio Ground-Truth Entity Recall (PHR): {presidio_results['phr']:.2f}%")
    print(f" OpenAI HumanEval Code Signature Integrity Rate      : {code_results['integrity_rate']:.2f}%")
    print(f" Average PII Redaction Latency                       : {presidio_results['latency']:.3f}s")
    print(f" Average Code Processing Latency                     : {code_results['latency']:.3f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
