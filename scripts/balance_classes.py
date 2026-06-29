"""
Cap over-represented classes in train.jsonl so the dominant:minority ratio
is at most 3:1 (configurable). Valid/test splits are left untouched.

Root cause: MolWt/MolLogP templates appear ~1800 times each while correct
minority APIs (QED, TPSA, Tanimoto) appear only ~300 times — a 7:1 ratio
that causes the model to hallucinate wrong API names at inference.

Usage:
    python scripts/balance_classes.py [--data data/sft_r5] [--ratio 3.0] [--dry-run]

Output:
    Overwrites data/sft_r5/train.jsonl with the balanced version.
    Prints before/after class counts.
"""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    type=Path, default=Path("data/sft_r5"))
    ap.add_argument("--ratio",   type=float, default=3.0,
                    help="max(class_count) / median(class_count) cap (default 3.0)")
    ap.add_argument("--seed",    type=int, default=51)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print analysis without modifying any files")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    train_path = args.data / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found — build the dataset first")

    with open(train_path) as f:
        examples = [json.loads(line) for line in f]

    before = Counter(ex.get("class", "unknown") for ex in examples)
    total_before = len(examples)

    counts = sorted(before.values())
    median = counts[len(counts) // 2]
    cap = math.ceil(median * args.ratio)

    print(f"Loaded {total_before} examples across {len(before)} classes")
    print(f"Median class size: {median}")
    print(f"Cap (ratio={args.ratio}): {cap} examples per class")
    print()

    # Group by class
    by_class: dict[str, list] = {}
    for ex in examples:
        cls = ex.get("class", "unknown")
        by_class.setdefault(cls, []).append(ex)

    # Cap and collect
    balanced = []
    capped_classes = []
    for cls, exs in sorted(by_class.items()):
        before_n = len(exs)
        if before_n > cap:
            rng.shuffle(exs)
            kept = exs[:cap]
            capped_classes.append((cls, before_n, cap))
        else:
            kept = exs
        balanced.append((cls, kept))

    after = {cls: len(exs) for cls, exs in balanced}
    all_balanced = [ex for _, exs in balanced for ex in exs]
    rng.shuffle(all_balanced)
    total_after = len(all_balanced)

    print("Classes that were capped:")
    if capped_classes:
        for cls, before_n, after_n in sorted(capped_classes, key=lambda x: -x[1]):
            print(f"  {cls:30s}  {before_n:5d} → {after_n:5d}  (−{before_n-after_n})")
    else:
        print("  None — dataset is already balanced within the ratio.")

    print(f"\nTotal: {total_before} → {total_after} examples"
          f" (removed {total_before - total_after})")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # Write balanced train.jsonl
    with open(train_path, "w") as f:
        for ex in all_balanced:
            f.write(json.dumps(ex) + "\n")

    print(f"\n✅ Wrote balanced train.jsonl → {train_path}")
    print("   valid.jsonl and test.jsonl are unchanged.")

    # Print final class distribution (top 20)
    after_counter = Counter(ex.get("class", "unknown") for ex in all_balanced)
    print("\nTop 20 classes after balancing:")
    for cls, n in after_counter.most_common(20):
        bar = "█" * int(30 * n / cap)
        print(f"  {cls:30s}  {n:5d}  {bar}")


if __name__ == "__main__":
    main()
