"""
label_pairs.py — Sample pairs from arb CSVs and produce a CSV for manual labeling.

Labels (3-class):
  0 = no match        — same topic but incompatible resolution conditions
  1 = partial match   — related conditions but different threshold / timeframe
  2 = true match      — same event + compatible resolution → genuine arb candidate

Heuristic pre-labels are written to the `label_hint` column to speed up review.
You override them in the `label` column (leave blank = needs review).

Usage:
    uv run label_pairs.py [--n 500] [--out data/labels/to_label.csv] [--seed 42]
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Heuristic weak-labeling
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> list[float]:
    """Pull every numeric value (int or decimal) out of a label string."""
    return [float(m) for m in re.findall(r"\d+\.?\d*", text)]


def _token_set(text: str) -> set[str]:
    tokens = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    stopwords = {"will", "the", "a", "an", "in", "on", "at", "to", "of",
                 "or", "and", "be", "is", "above", "below", "this", "that",
                 "by", "for", "with", "from", "than", "least", "most"}
    return {t for t in tokens if t not in stopwords}


def heuristic_label(row: pd.Series) -> tuple[int, str]:
    """
    Return (label_hint, reason).
    Conservative: only assigns 0 or 2 when the signal is clear.
    Everything else stays -1 (needs human review).
    """
    kl = str(row["kalshi_label"]).strip()
    pl = str(row["poly_label"]).strip()

    k_nums = set(_extract_numbers(kl))
    p_nums = set(_extract_numbers(pl))
    shared_tokens = _token_set(kl) & _token_set(pl)
    jaccard = len(shared_tokens) / max(len(_token_set(kl) | _token_set(pl)), 1)

    price_delta = abs(row["kalshi_yes"] - (100.0 - row["poly_yes"]))

    # ── Strong false-positive signals → class 0 ──────────────────────────
    # Opposite conditions: one asks "win", the other "eliminated/voted off"
    opposite_pairs = [
        ({"win", "winner"}, {"eliminat", "voted off", "evict"}),
        ({"above", "higher", "over"}, {"below", "under", "lower"}),
    ]
    kl_lower, pl_lower = kl.lower(), pl.lower()
    for pos_kw, neg_kw in opposite_pairs:
        k_has_pos = any(w in kl_lower for w in pos_kw)
        k_has_neg = any(w in kl_lower for w in neg_kw)
        p_has_pos = any(w in pl_lower for w in pos_kw)
        p_has_neg = any(w in pl_lower for w in neg_kw)
        if (k_has_pos and p_has_neg) or (k_has_neg and p_has_pos):
            return 0, "opposite conditions"

    # Numeric thresholds present in both labels but they don't overlap
    if k_nums and p_nums and k_nums.isdisjoint(p_nums):
        # Only flag if the numbers are meaningfully different (not just formatting)
        k_max, p_max = max(k_nums), max(p_nums)
        if abs(k_max - p_max) / max(p_max, 1) > 0.25:
            return 1, f"threshold mismatch ({k_max} vs {p_max})"

    # ── Strong true-match signals → class 2 ──────────────────────────────
    # Very high similarity + near-zero price delta (proper arb pricing)
    if row["similarity"] >= 0.96 and price_delta <= 3.0:
        return 2, f"high sim ({row['similarity']:.3f}) + low price delta ({price_delta:.1f}¢)"

    # Same numeric thresholds + high token overlap
    if k_nums and p_nums and not k_nums.isdisjoint(p_nums) and jaccard >= 0.50:
        return 2, f"matching thresholds + jaccard {jaccard:.2f}"

    return -1, "needs review"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def load_all_pairs(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(f"{data_dir}/arb_*.csv"))
    if not files:
        sys.exit(f"No arb_*.csv files found in {data_dir}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["kalshi_id", "poly_id"])
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows proportionally from each similarity bucket."""
    bins = [0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1.01]
    bucket_labels = ["0.82-0.85", "0.85-0.88", "0.88-0.91",
                     "0.91-0.94", "0.94-0.97", "0.97-1.00"]
    df = df.copy()
    df["_bucket"] = pd.cut(df["similarity"], bins=bins, labels=bucket_labels)

    bucket_counts = df["_bucket"].value_counts()
    total = bucket_counts.sum()
    samples = []
    for bucket, count in bucket_counts.items():
        k = max(1, round(n * count / total))
        k = min(k, count)
        samples.append(df[df["_bucket"] == bucket].sample(n=k, random_state=seed))

    sampled = pd.concat(samples).sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled.drop(columns=["_bucket"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a labeling CSV from arb data.")
    parser.add_argument("--n", type=int, default=500, help="Number of pairs to sample")
    parser.add_argument("--data-dir", default="data/arb", help="Directory with arb_*.csv files")
    parser.add_argument("--out", default="data/labels/to_label.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading pairs from {args.data_dir}...")
    df = load_all_pairs(args.data_dir)
    print(f"  {len(df)} unique pairs loaded.")

    print(f"Sampling {args.n} pairs (stratified by similarity)...")
    sample = stratified_sample(df, args.n, args.seed)

    print("Applying heuristic pre-labels...")
    hints = sample.apply(heuristic_label, axis=1)
    sample["label_hint"] = [h[0] for h in hints]
    sample["hint_reason"] = [h[1] for h in hints]
    sample["label"] = ""  # human fills this in

    hint_counts = sample["label_hint"].value_counts().sort_index()
    print(f"  label -1 (needs review): {hint_counts.get(-1, 0)}")
    print(f"  label  0 (no match):     {hint_counts.get(0, 0)}")
    print(f"  label  1 (partial):      {hint_counts.get(1, 0)}")
    print(f"  label  2 (true match):   {hint_counts.get(2, 0)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "label", "label_hint", "hint_reason",
        "similarity", "edge_cents", "strategy",
        "kalshi_id", "kalshi_label", "kalshi_yes", "kalshi_no",
        "poly_id", "poly_label", "poly_yes", "poly_no",
    ]
    sample[cols].to_csv(out_path, index=False)
    print(f"\nSaved {len(sample)} rows → {out_path}")
    print("\nNext steps:")
    print("  1. Open the CSV (Excel / Google Sheets / any editor)")
    print("  2. For each row, check label_hint — if it looks right, copy it to 'label'")
    print("  3. For rows with label_hint=-1, manually set label to 0, 1, or 2")
    print("  4. Save and run:  uv run src/ml/train.py --labels data/labels/to_label.csv")


if __name__ == "__main__":
    main()
