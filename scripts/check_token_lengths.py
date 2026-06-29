"""
Measure token-length distribution of sft_r5/train.jsonl.
Outputs p50/p90/p95/p99/max and a histogram.
Used to validate max_seq_length before R5 training.

Usage:
    python scripts/check_token_lengths.py [--data data/sft_r5] [--split train]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from transformers import AutoTokenizer
    HAS_HF = True
except ImportError:
    HAS_HF = False


def _chat_to_text(messages: list[dict]) -> str:
    """Rough token-count proxy: concatenate role+content."""
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  type=Path, default=Path("data/sft_r5"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-32B-Instruct-4bit",
                    help="HuggingFace tokenizer to use (optional)")
    args = ap.parse_args()

    jsonl = args.data / f"{args.split}.jsonl"
    if not jsonl.exists():
        sys.exit(f"File not found: {jsonl}")

    tokenizer = None
    if HAS_HF:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            print(f"Using tokenizer: {args.model}")
        except Exception as e:
            print(f"Warning: could not load tokenizer ({e}), falling back to whitespace split")
    else:
        print("transformers not installed — using whitespace-split proxy (±10% accurate)")

    lengths = []
    with open(jsonl) as f:
        for line in f:
            ex = json.loads(line)
            msgs = ex.get("messages", [])
            text = _chat_to_text(msgs)
            if tokenizer:
                n = len(tokenizer.encode(text))
            else:
                n = len(text.split())
            lengths.append(n)

    lengths = np.array(lengths)
    print(f"\nToken lengths over {len(lengths)} examples ({args.split} split):")
    print(f"  min   : {lengths.min()}")
    print(f"  p50   : {int(np.percentile(lengths, 50))}")
    print(f"  p90   : {int(np.percentile(lengths, 90))}")
    print(f"  p95   : {int(np.percentile(lengths, 95))}")
    print(f"  p99   : {int(np.percentile(lengths, 99))}")
    print(f"  max   : {lengths.max()}")
    print(f"  mean  : {lengths.mean():.0f}")

    # Histogram
    bins = [0, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 999999]
    labels = ["<256", "256-512", "512-768", "768-1k", "1k-1.5k",
              "1.5k-2k", "2k-3k", "3k-4k", "4k-6k", ">6k"]
    counts, _ = np.histogram(lengths, bins=bins)
    print("\nDistribution:")
    for label, count in zip(labels, counts):
        bar = "█" * int(50 * count / len(lengths))
        pct = 100 * count / len(lengths)
        print(f"  {label:10s}  {bar:<50s}  {count:5d} ({pct:4.1f}%)")

    p99 = int(np.percentile(lengths, 99))
    if p99 <= 2048:
        print("\n✅ p99 fits within 2048 — max_seq_length=2048 is sufficient.")
    elif p99 <= 3072:
        print("\n✅ p99 fits within 3072 — max_seq_length=3072 (R5 config) is sufficient.")
    elif p99 <= 4096:
        print("\n⚠️  p99 exceeds 3072 — consider max_seq_length=4096 in train_config_r5.yaml.")
    else:
        print("\n❌ p99 exceeds 4096 — investigate long examples before training.")


if __name__ == "__main__":
    main()
