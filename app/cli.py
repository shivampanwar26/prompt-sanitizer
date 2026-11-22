import argparse
import os
import sys
import time

from sanitizer import PromptSanitizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prompt Privacy Sanitizer"
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Prompt text to sanitize",
    )
    parser.add_argument("--file", "-f", type=str, help="Path to file containing prompts")
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Enable evaluation metrics (perplexity, PHR). Adds ~30%% latency.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device; auto uses CUDA when available (default: auto).",
    )
    parser.add_argument(
        "--image",
        nargs="+",
        metavar="PATH",
        help="Sanitize one or more images instead of text (faces, in-image text, metadata).",
    )
    parser.add_argument("--out-dir", default="sanitized_images", help="Where --image writes its output")
    parser.add_argument("--no-face-redact", action="store_true", help="Skip face detection/redaction for --image")
    parser.add_argument("--no-text-redact", action="store_true", help="Skip in-image text detection/redaction for --image")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="For --image: also run OCR'd text through the full contextual pipeline, not just pattern/NER",
    )
    return parser.parse_args()


def run_image_mode(args) -> None:
    from sanitizer import ImageSanitizer

    os.makedirs(args.out_dir, exist_ok=True)
    image_sanitizer = ImageSanitizer(redact_faces=not args.no_face_redact, redact_text=not args.no_text_redact)
    prompt_sanitizer = PromptSanitizer(device=args.device) if args.deep else None

    print("\n" + "=" * 80)
    print(" Image Privacy Sanitizer ")
    print("=" * 80)

    for path in args.image:
        t0 = time.time()
        result = image_sanitizer.sanitize_file(path, deep=args.deep, prompt_sanitizer=prompt_sanitizer)
        elapsed = time.time() - t0

        out_path = os.path.join(args.out_dir, os.path.basename(path))
        result.save(out_path)

        print(f"\n{path} -> {out_path}  ({elapsed:.3f}s)")
        print(f"  Faces redacted        : {result.faces_redacted}")
        print(f"  Text regions redacted : {result.text_regions_redacted}"
              + (f"  [{', '.join(result.pii_kinds_found)}]" if result.pii_kinds_found else ""))
        print(f"  Metadata removed      : {', '.join(result.metadata_removed) or '(none found)'}")
        for warning in result.warnings:
            print(f"  Warning: {warning}")


def main():
    args = parse_args()

    if args.image:
        run_image_mode(args)
        return

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
    sanitizer = PromptSanitizer(device=args.device)
    session = sanitizer.new_session()
    t_model_load = time.time() - t0
    print(f"Inference device: {sanitizer._analyzer.device}")

    print("\n" + "=" * 80)
    print(" Prompt Privacy Sanitizer ")
    print("=" * 80)

    for idx, prompt in enumerate(prompts, start=1):
        if len(prompts) > 1:
            print(f"\n--- Prompt [{idx}/{len(prompts)}] ---")

        t_start = time.time()
        result = sanitizer.sanitize(
            prompt,
            evaluate=args.eval,
            session=session
        )
        t_infer = time.time() - t_start

        print("\nOriginal Prompt:")
        print(f"  {prompt}")

        print("\nSanitized Prompt:")
        print(f"  {result.text}")

        if args.eval:
            print("\nPrompt Privacy & Readability Evaluation:")
            print(f"  Average Self-Information (H_q) : {result.H_q:.4f} bits")
            print(f"  Dynamic Protection Ratio (gamma_q): {result.gamma_q:.4f}")
            if result.original_perplexity is not None:
                print(f"  Original Perplexity (PPL)    : {result.original_perplexity:.4f}")
            if result.perplexity is not None:
                print(f"  Sanitized Perplexity (PPL)   : {result.perplexity:.4f}")
            if result.phr is not None:
                print(f"  Privacy Hiding Rate (PHR)    : {result.phr:.2f}%")
        print(f"  Inference Latency            : {t_infer:.3f} seconds")

        if result.already_masked:
            print("\nAlready masked:", ", ".join(span["word"] for span in result.already_masked))
        print("\nDesensitized Word Spans:")
        if not result.selected_words:
            print("  (No sensitive words or PII detected)")
        else:
            print(
                f"  {'Original':<22} -> {'Replacement':<22} | "
                f"{'Type':<16} | {'Utility K_w':<11} | {'Privacy O_w':<11} | {'Self-Info I_w':<12}"
            )
            print("  " + "-" * 110)
            for item in result.selected_words:
                print(
                    f"  {item['word']!r:<22} -> {item['replacement']!r:<22} | "
                    f"{item['pos_tag']:<16} | "
                    f"{item['importance']:<11.3f} | "
                    f"{item['privacy']:<11.3f} | "
                    f"{item['raw_privacy']:<8.3f} bits"
                )

    print("\n" + "=" * 80)
    print(f"Total time (Model Load: {t_model_load:.2f}s)")


if __name__ == "__main__":
    main()
