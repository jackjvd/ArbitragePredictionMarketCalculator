"""
src/ml/features.py — Feature extraction for the 3-class pair classifier.

Each row in the arb CSV becomes a feature vector with:
  - Structural price features (price_delta, edge_cents, kalshi/poly prices)
  - Text overlap features (jaccard, threshold match, length ratio)
  - Similarity score from SBERT
  - Kalshi market category (from market ID prefix)

Optionally adds cross-encoder scores if a model is available.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"\d+\.?\d*", text)]


def _token_set(text: str) -> set[str]:
    stopwords = {"will", "the", "a", "an", "in", "on", "at", "to", "of",
                 "or", "and", "be", "is", "above", "below", "this", "that",
                 "by", "for", "with", "from", "than", "least", "most",
                 "score", "reach", "price", "week", "month", "year"}
    tokens = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    return {t for t in tokens if t not in stopwords}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _has_any(text: str, words: list[str]) -> int:
    t = text.lower()
    return int(any(w in t for w in words))


def _kalshi_category(market_id: str) -> str:
    """Extract the event category prefix from a Kalshi market ID.
    e.g. 'KXDOGED-26MAR0905-T0.05' → 'KXDOGED'
    """
    return market_id.split("-")[0] if "-" in market_id else market_id


# ---------------------------------------------------------------------------
# Core feature builder
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Price features
    "price_delta",
    "kalshi_yes",
    "kalshi_no",
    "poly_yes",
    "poly_no",
    "edge_cents",
    "log_edge_cents",
    # Similarity
    "similarity",
    # Text overlap
    "jaccard",
    "threshold_match",
    "threshold_delta",
    "kalshi_num_count",
    "poly_num_count",
    "label_length_ratio",
    "char_overlap_ratio",
    # Semantic signals
    "k_has_win",
    "k_has_eliminated",
    "p_has_win",
    "p_has_eliminated",
    "k_has_above",
    "k_has_below",
    "p_has_above",
    "p_has_below",
    "k_has_weekly",
    "p_has_weekly",
    # Category (encoded)
    "kalshi_category_enc",
]


def build_features(df: pd.DataFrame, category_encoder: Optional[LabelEncoder] = None) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Build the feature matrix from an arb DataFrame.

    Returns (feature_df, fitted_category_encoder).
    Pass a pre-fitted encoder when transforming test/inference data.
    """
    f = pd.DataFrame(index=df.index)

    # ── Price features ────────────────────────────────────────────────────
    f["kalshi_yes"] = df["kalshi_yes"].astype(float)
    f["kalshi_no"] = df["kalshi_no"].astype(float)
    f["poly_yes"] = df["poly_yes"].astype(float)
    f["poly_no"] = df["poly_no"].astype(float)
    f["edge_cents"] = df["edge_cents"].astype(float)
    f["log_edge_cents"] = np.log1p(f["edge_cents"])

    # price_delta: how far apart the implied YES prices are across platforms
    # A true arb should have kalshi_yes ≈ 100 - poly_yes (or vice versa)
    f["price_delta"] = (f["kalshi_yes"] - (100.0 - f["poly_yes"])).abs()

    # ── Similarity ────────────────────────────────────────────────────────
    f["similarity"] = df["similarity"].astype(float)

    # ── Text overlap ─────────────────────────────────────────────────────
    k_labels = df["kalshi_label"].fillna("").astype(str)
    p_labels = df["poly_label"].fillna("").astype(str)

    k_tokens = k_labels.map(_token_set)
    p_tokens = p_labels.map(_token_set)
    f["jaccard"] = [_jaccard(k, p) for k, p in zip(k_tokens, p_tokens)]

    k_nums = k_labels.map(_extract_numbers)
    p_nums = p_labels.map(_extract_numbers)
    f["kalshi_num_count"] = k_nums.map(len)
    f["poly_num_count"] = p_nums.map(len)

    def _threshold_match(kn: list[float], pn: list[float]) -> int:
        if not kn or not pn:
            return 0
        return int(not set(kn).isdisjoint(set(pn)))

    def _threshold_delta(kn: list[float], pn: list[float]) -> float:
        """Max absolute difference between closest numeric pair across labels."""
        if not kn or not pn:
            return -1.0
        return min(abs(k - p) for k in kn for p in pn)

    f["threshold_match"] = [_threshold_match(k, p) for k, p in zip(k_nums, p_nums)]
    f["threshold_delta"] = [_threshold_delta(k, p) for k, p in zip(k_nums, p_nums)]

    f["label_length_ratio"] = k_labels.str.len() / p_labels.str.len().replace(0, 1)

    # Character-level overlap: fraction of kalshi chars that appear in poly
    def _char_overlap(a: str, b: str) -> float:
        a_chars = set(a.lower())
        b_chars = set(b.lower())
        if not a_chars:
            return 0.0
        return len(a_chars & b_chars) / len(a_chars)

    f["char_overlap_ratio"] = [_char_overlap(k, p) for k, p in zip(k_labels, p_labels)]

    # ── Semantic keyword signals ──────────────────────────────────────────
    f["k_has_win"] = k_labels.map(lambda t: _has_any(t, ["win", "winner", "wins"]))
    f["k_has_eliminated"] = k_labels.map(lambda t: _has_any(t, ["eliminat", "voted off", "evict", "out"]))
    f["p_has_win"] = p_labels.map(lambda t: _has_any(t, ["win", "winner", "wins"]))
    f["p_has_eliminated"] = p_labels.map(lambda t: _has_any(t, ["eliminat", "voted off", "evict", "out"]))
    f["k_has_above"] = k_labels.map(lambda t: _has_any(t, ["above", "over", "higher", "at least", "or more"]))
    f["k_has_below"] = k_labels.map(lambda t: _has_any(t, ["below", "under", "lower", "at most", "or less"]))
    f["p_has_above"] = p_labels.map(lambda t: _has_any(t, ["above", "over", "higher", "at least", "or more", "reach"]))
    f["p_has_below"] = p_labels.map(lambda t: _has_any(t, ["below", "under", "lower", "at most", "or less"]))
    f["k_has_weekly"] = k_labels.map(lambda t: _has_any(t, ["week", "daily", "today", "tonight"]))
    f["p_has_weekly"] = p_labels.map(lambda t: _has_any(t, ["week", "daily", "today", "tonight"]))

    # ── Kalshi category ───────────────────────────────────────────────────
    categories = df["kalshi_id"].fillna("").map(_kalshi_category)
    if category_encoder is None:
        category_encoder = LabelEncoder()
        f["kalshi_category_enc"] = category_encoder.fit_transform(categories)
    else:
        # Handle unseen categories gracefully
        known = set(category_encoder.classes_)
        safe = categories.map(lambda c: c if c in known else "__unknown__")
        if "__unknown__" not in known:
            category_encoder.classes_ = np.append(category_encoder.classes_, "__unknown__")
        f["kalshi_category_enc"] = category_encoder.transform(safe)

    return f[FEATURE_COLS], category_encoder


# ---------------------------------------------------------------------------
# Optional: cross-encoder scoring (much slower, much more accurate)
# ---------------------------------------------------------------------------

def add_cross_encoder_scores(df: pd.DataFrame, feature_df: pd.DataFrame, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> pd.DataFrame:
    """
    Add a cross-encoder score column to feature_df.
    Only call this if you have sentence-transformers installed and want
    higher-quality features at the cost of inference time.
    """
    from sentence_transformers import CrossEncoder

    print(f"Loading cross-encoder ({model_name})...")
    model = CrossEncoder(model_name)

    pairs = list(zip(
        df["kalshi_label"].fillna("").astype(str),
        df["poly_label"].fillna("").astype(str),
    ))
    print(f"Scoring {len(pairs)} pairs...")
    scores = model.predict(pairs, show_progress_bar=True)
    feature_df = feature_df.copy()
    feature_df["cross_encoder_score"] = scores
    return feature_df
