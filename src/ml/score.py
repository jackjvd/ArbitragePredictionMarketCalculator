"""
src/ml/score.py — Score new arb pairs with the trained classifier.

Adds three columns to the input CSV:
  pred_label       — predicted class (0, 1, 2)
  pred_label_name  — human-readable class name
  pred_prob_0/1/2  — class probabilities

Usage:
    uv run src/ml/score.py --model models/pair_classifier.pkl --data data/arb/arb_LATEST.csv
    uv run src/ml/score.py --model models/pair_classifier.pkl --data data/arb/arb_LATEST.csv --min-class 2
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

from src.ml.features import build_features

LABEL_NAMES = {0: "no_match", 1: "partial_match", 2: "true_match"}


def filter_true_matches(df: pd.DataFrame, min_prob: float = 0.0, prob_col: str = "pred_prob_2") -> pd.DataFrame:
    """Return only rows predicted as true matches, optionally by confidence.

    Args:
        df: Scored dataframe containing `pred_label` and probability columns.
        min_prob: Minimum probability threshold for class 2.
        prob_col: Probability column for class 2 confidence.
    """
    if "pred_label" not in df.columns:
        sys.exit("Missing required column: pred_label")
    if prob_col not in df.columns:
        sys.exit(f"Missing required column: {prob_col}")

    return df[(df["pred_label"] == 2) & (df[prob_col] >= min_prob)].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to pair_classifier.pkl")
    parser.add_argument("--data", required=True, help="arb CSV to score")
    parser.add_argument("--out", default=None, help="Output path (default: input_scored.csv)")
    parser.add_argument("--min-class", type=int, default=None,
                        help="Only output rows with pred_label >= this value (e.g. 2 = true matches only)")
    parser.add_argument("--true-only", action="store_true",
                        help="Keep only rows with pred_label=2 (true_match)")
    parser.add_argument("--min-true-prob", type=float, default=0.0,
                        help="When --true-only is used, require pred_prob_2 >= this threshold")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    with open(args.model, "rb") as fh:
        artifacts = pickle.load(fh)

    model = artifacts["model"]
    cat_enc = artifacts["category_encoder"]
    feature_names = artifacts["feature_names"]

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"  {len(df)} rows")

    print("Extracting features...")
    X, _ = build_features(df, category_encoder=cat_enc)
    X = X[feature_names]

    print("Scoring...")
    probs = model.predict_proba(X)
    preds = model.predict(X)

    df["pred_label"] = preds
    df["pred_label_name"] = pd.Series(preds).map(LABEL_NAMES)
    for i in range(probs.shape[1]):
        df[f"pred_prob_{i}"] = probs[:, i].round(4)

    if args.true_only:
        before = len(df)
        df = filter_true_matches(df, min_prob=args.min_true_prob)
        print(f"Filtered to true_match (pred_label=2) with pred_prob_2 >= {args.min_true_prob}: {before} → {len(df)} rows")

    if args.min_class is not None:
        before = len(df)
        df = df[df["pred_label"] >= args.min_class]
        print(f"Filtered to pred_label >= {args.min_class}: {before} → {len(df)} rows")

    print("\nPrediction distribution:")
    print(df["pred_label_name"].value_counts().to_string())

    out_path = args.out or str(Path(args.data).with_suffix("")) + "_scored.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
