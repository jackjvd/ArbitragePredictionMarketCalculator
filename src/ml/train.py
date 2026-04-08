"""
src/ml/train.py — Train a 3-class XGBoost classifier on labeled arb pairs.

Classes:
  0 = no match        (same topic, incompatible resolution conditions)
  1 = partial match   (related but different threshold / timeframe)
  2 = true match      (same event, compatible conditions — genuine arb)

Usage:
    uv run src/ml/train.py --labels data/labels/to_label.csv
    uv run src/ml/train.py --labels data/labels/to_label.csv --cross-encoder
    uv run src/ml/train.py --labels data/labels/to_label.csv --eval-only --model models/pair_classifier.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

try:
    import xgboost as xgb
except ImportError:
    sys.exit("xgboost is required: uv add xgboost")

from src.ml.features import FEATURE_COLS, add_cross_encoder_scores, build_features

LABEL_NAMES = {0: "no_match", 1: "partial_match", 2: "true_match"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_labeled(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"label", "kalshi_label", "poly_label", "similarity",
                 "edge_cents", "kalshi_yes", "kalshi_no",
                 "poly_yes", "poly_no", "kalshi_id"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Missing columns in labeled CSV: {missing}")

    before = len(df)
    df = df[df["label"].apply(lambda x: str(x).strip() not in ("", "-1", "nan"))]
    df["label"] = df["label"].astype(int)
    after = len(df)
    print(f"Loaded {before} rows, {after} labeled ({before - after} skipped).")

    if df["label"].nunique() < 2:
        sys.exit("Need at least 2 classes in labeled data to train.")

    print("Class distribution:")
    for cls, name in LABEL_NAMES.items():
        count = (df["label"] == cls).sum()
        print(f"  {cls} ({name}): {count}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(scale_pos: dict | None = None) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def compute_class_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().sort_index()
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return y.map(weights).values


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def print_report(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> None:
    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    labels = sorted(set(y_true) | set(y_pred))
    names = [LABEL_NAMES.get(l, str(l)) for l in labels]
    print(f"\n{'='*60}")
    if prefix:
        print(prefix)
    print(classification_report(y_true, y_pred, labels=labels, target_names=names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix (rows=actual, cols=predicted):")
    header = "       " + "  ".join(f"{n[:8]:>8}" for n in names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {names[i][:8]:>8}  " + "  ".join(f"{v:>8d}" for v in row))


def feature_importance_table(model: xgb.XGBClassifier, feature_names: list[str], top_n: int = 15) -> pd.DataFrame:
    scores = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": scores})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to labeled CSV")
    parser.add_argument("--out-dir", default="models", help="Where to save model artifacts")
    parser.add_argument("--cross-encoder", action="store_true",
                        help="Add cross-encoder scores (slow but more accurate)")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--eval-only", action="store_true",
                        help="Load existing model and evaluate, skip training")
    parser.add_argument("--model", default=None, help="Path to saved model (for --eval-only)")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────
    df = load_labeled(args.labels)
    y = df["label"].values

    # ── Features ──────────────────────────────────────────────────────────
    print("\nExtracting features...")
    X, cat_enc = build_features(df)

    if args.cross_encoder:
        print("Adding cross-encoder scores (this may take a few minutes)...")
        X = add_cross_encoder_scores(df, X)

    feature_names = list(X.columns)
    print(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Eval-only mode ────────────────────────────────────────────────────
    if args.eval_only:
        model_path = args.model or str(out_dir / "pair_classifier.pkl")
        with open(model_path, "rb") as fh:
            artifacts = pickle.load(fh)
        model = artifacts["model"]
        y_pred = model.predict(X[artifacts["feature_names"]])
        print_report(y, y_pred, prefix=f"Evaluation on {len(y)} labeled samples")
        return

    # ── Cross-validation ──────────────────────────────────────────────────
    print(f"\nRunning {args.cv_folds}-fold stratified cross-validation...")
    model_cv = build_model()
    sample_weight = compute_class_weights(df["label"])

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    try:
        # Older scikit-learn versions expect fit_params.
        y_cv_pred = cross_val_predict(
            model_cv, X, y,
            cv=cv,
            fit_params={"sample_weight": sample_weight},
            n_jobs=1,
        )
    except TypeError:
        # Newer scikit-learn versions pass estimator kwargs via params.
        y_cv_pred = cross_val_predict(
            model_cv, X, y,
            cv=cv,
            params={"sample_weight": sample_weight},
            n_jobs=1,
        )
    print_report(y, y_cv_pred, prefix=f"{args.cv_folds}-fold CV results")

    # ── Final model on full data ───────────────────────────────────────────
    print("\nTraining final model on full labeled dataset...")
    model = build_model()
    model.fit(X, y, sample_weight=sample_weight)

    # In-sample sanity check
    y_train_pred = model.predict(X)
    print_report(y, y_train_pred, prefix="Train set (in-sample, for sanity)")

    # ── Feature importance ────────────────────────────────────────────────
    print("\nTop feature importances:")
    fi = feature_importance_table(model, feature_names)
    print(fi.to_string(index=False))

    # ── Save artifacts ────────────────────────────────────────────────────
    artifacts = {
        "model": model,
        "category_encoder": cat_enc,
        "feature_names": feature_names,
        "label_names": LABEL_NAMES,
        "cv_report": classification_report(y, y_cv_pred, output_dict=True),
    }
    model_path = out_dir / "pair_classifier.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(artifacts, fh)

    report_path = out_dir / "cv_report.json"
    with open(report_path, "w") as fh:
        json.dump(artifacts["cv_report"], fh, indent=2)

    print(f"\nSaved model   → {model_path}")
    print(f"Saved report  → {report_path}")
    print("\nNext: use the model to score new pairs:")
    print("  uv run src/ml/score.py --model models/pair_classifier.pkl --data data/arb/arb_LATEST.csv")


if __name__ == "__main__":
    main()
