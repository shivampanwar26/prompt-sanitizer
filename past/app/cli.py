import argparse
import sys
import time
from sanitizer import PromptSanitizer, SanitizerConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="ProSan Prompt Privacy Sanitizer — Clean & Simple CLI"
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Prompt text to sanitize (e.g. python -m app.cli 'My name is John')",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to file containing prompts (one per line)",
    )
    parser.add_argument(
        "--history",
        nargs="+",
        help="Optional multi-turn conversation context history",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Enable evaluation metrics (perplexity, PHR). Adds ~30% latency.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    prompts = []
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.text:
        prompts = [" ".join(args.text)]
    else:
        user_input = input("Enter prompt: ")
        if user_input.strip():
            prompts = [user_input.strip()]

    if not prompts:
        print("Error: No prompt provided.")
        sys.exit(1)

    t0 = time.time()
    config = SanitizerConfig.load_from_yaml()
    sanitizer = PromptSanitizer(config=config)
    t_model_load = time.time() - t0

    print("\n" + "=" * 80)
    print(" ProSan Prompt Privacy Sanitizer ")
    print("=" * 80)

    for idx, prompt in enumerate(prompts, start=1):
        if len(prompts) > 1:
            print(f"\n--- Prompt [{idx}/{len(prompts)}] ---")

        t_start = time.time()
        result = sanitizer.sanitize(prompt, history=args.history, evaluate=args.eval)
        t_infer = time.time() - t_start

        print("\nOriginal Prompt:")
        print(f"  {prompt}")

        print("\nSanitized Prompt:")
        print(f"  {result.text}")

        if args.eval:
            print(f"\nPrompt Privacy & Readability Evaluation:")
            print(f"  Average Self-Information (H_q) : {result.H_q:.4f} bits")
            print(f"  Dynamic Protection Ratio (gamma_q): {result.gamma_q:.4f}")
            if result.original_perplexity is not None:
                print(f"  Original Perplexity (PPL)    : {result.original_perplexity:.4f}")
            if result.perplexity is not None:
                print(f"  Sanitized Perplexity (PPL)   : {result.perplexity:.4f}")
            if result.phr is not None:
                print(f"  Privacy Hiding Rate (PHR)    : {result.phr:.2f}%")
        print(f"  GPU Inference Latency        : {t_infer:.3f} seconds")

        print("\nDesensitized Word Spans:")
        if not result.selected_words:
            print("  (No sensitive words or PII detected)")
        else:
            print(
                f"  {'Original':<18} -> {'Replacement':<16} | "
                f"{'POS':<10} | {'Utility K_w':<11} | {'Privacy O_w':<11} | {'Self-Info I_w':<12}"
            )
            print("  " + "-" * 88)
            for item in result.selected_words:
                rep = item.get("replacement") or "N/A"
                print(
                    f"  {item['word']!r:<18} -> {rep!r:<16} | "
                    f"{item['pos_tag']:<10} | "
                    f"{item['importance']:<11.3f} | "
                    f"{item['privacy']:<11.3f} | "
                    f"{item['raw_privacy']:<12.3f} bits"
                )

    print("\n" + "=" * 80)
    print(f"Total time (Model Load: {t_model_load:.2f}s | Device: {config.device.upper()})")


if __name__ == "__main__":
    main()
