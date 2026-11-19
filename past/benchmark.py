"""Benchmark: Our Enhanced Pipeline vs Original ProSan Approach.

Compares: latency, perplexity preservation, PHR, and replacement quality.
"""
import time
import torch
from sanitizer import PromptSanitizer, SanitizerConfig

PROMPTS = [
    "Jack was born on July 2nd and has had a fever of 103°F for 2 days. What steps should be taken?",
    "Sarah was born on March 14th and lives at 42 Oak Street, Seattle. What should she do about her recurring headaches?",
    "My email is john@example.com and my phone is 555-123-4567. I need help with chest pain.",
    "Contact Dr. Emily at emily.chen@hospital.org for the lab results from 05/12/2024.",
    "[...] connection = psycopg2.connect(host=114.165.14.1, database=production, user=Estrella, password=>@]@x5<pSA) [...] The above code raises an ImportError. Please fix it.",
]

def run_benchmark():
    config = SanitizerConfig.load_from_yaml()
    sanitizer = PromptSanitizer(config=config)

    # Warm up models
    sanitizer.sanitize("warm up test", evaluate=True)

    print("=" * 90)
    print(" BENCHMARK: Enhanced ProSan Pipeline")
    print("=" * 90)

    total_latency = 0
    results = []

    for i, prompt in enumerate(PROMPTS, 1):
        t0 = time.time()
        result = sanitizer.sanitize(prompt, evaluate=True)
        latency = time.time() - t0
        total_latency += latency

        ppl_ratio = (result.perplexity / result.original_perplexity
                     if result.original_perplexity and result.original_perplexity > 0
                     else None)

        results.append({
            "prompt_id": i,
            "latency": latency,
            "orig_ppl": result.original_perplexity,
            "san_ppl": result.perplexity,
            "ppl_ratio": ppl_ratio,
            "phr": result.phr,
            "n_replacements": len(result.selected_words),
            "H_q": result.H_q,
        })

        print(f"\n--- Prompt {i} ---")
        print(f"  Original:  {prompt[:80]}...")
        print(f"  Sanitized: {result.text[:80]}...")
        print(f"  Latency:   {latency:.3f}s")
        print(f"  PPL orig:  {result.original_perplexity}")
        print(f"  PPL san:   {result.perplexity}")
        print(f"  PPL ratio: {ppl_ratio:.4f}" if ppl_ratio else "  PPL ratio: N/A")
        print(f"  PHR:       {result.phr}%")
        print(f"  Replaced:  {len(result.selected_words)} words")

        for item in result.selected_words:
            rep = item.get("replacement", "N/A")
            print(f"    '{item['word']}' -> '{rep}' [{item['pos_tag']}]")

    # Summary
    avg_latency = total_latency / len(PROMPTS)
    avg_ppl_ratio = sum(r["ppl_ratio"] for r in results if r["ppl_ratio"]) / len([r for r in results if r["ppl_ratio"]])
    avg_phr = sum(r["phr"] for r in results if r["phr"] is not None) / len([r for r in results if r["phr"] is not None])

    print("\n" + "=" * 90)
    print(" SUMMARY")
    print("=" * 90)
    print(f"  Avg Latency (with eval):    {avg_latency:.3f}s")
    print(f"  Avg PPL Ratio (san/orig):   {avg_ppl_ratio:.4f}  (closer to 1.0 = better utility)")
    print(f"  Avg PHR:                    {avg_phr:.2f}%  (higher = better privacy)")
    print(f"  Total prompts tested:       {len(PROMPTS)}")
    print()
    print("  ProSan Paper Comparison:")
    print("  ─────────────────────────────────────────────────────────")
    print(f"  │ Metric              │ ProSan Paper │ Our Enhanced  │")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  │ Importance Method   │ Gradient     │ Attention     │")
    print(f"  │ Similarity          │ WordNet      │ Embedding     │")
    print(f"  │ Scoring             │ Eq.9 only    │ MLM × Eq.9   │")
    print(f"  │ Batching            │ Sequential   │ Batched       │")
    print(f"  │ Precision           │ FP32         │ FP16          │")
    print(f"  │ PII Detection       │ None         │ Regex+NER     │")
    print(f"  │ Avg Latency         │ ~2.5-3s      │ {avg_latency:.3f}s        │")
    print(f"  │ PPL Ratio           │ ~1.5-2.5     │ {avg_ppl_ratio:.4f}       │")
    print(f"  │ PHR                 │ ~90-95%      │ {avg_phr:.2f}%      │")
    print(f"  ─────────────────────────────────────────────────────────")
    print("=" * 90)


if __name__ == "__main__":
    run_benchmark()
