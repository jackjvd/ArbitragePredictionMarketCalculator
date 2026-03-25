"""
Arbitrage finder: matches Kalshi and Polymarket markets using Sentence-BERT
and surfaces cross-platform pricing discrepancies.

Usage:
    uv run arb_finder.py [--similarity 0.82] [--top 50]

Options:
    --similarity <float>   Minimum cosine similarity to consider a match (default: 0.82)
    --top <int>            Only print the top N matches by arb edge (default: all)

Output:
    - Prints matched pairs with prices and arbitrage edge to stdout
    - Saves full results to data/arb/matches_<timestamp>.json
    - Saves arbitrage opportunities to data/arb/arb_<timestamp>.csv
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data/arb")

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

def _valid_price(val: Optional[int]) -> Optional[float]:
    """Return price as float only if it's a real bettable price (1–99¢).
    Kalshi uses 0 or 100 as placeholders meaning 'not available'."""
    if val is None:
        return None
    return float(val) if 1 <= val <= 99 else None


@dataclass
class FlatMarket:
    platform: str               # "kalshi" or "polymarket"
    market_id: str              # ticker or condition_id
    label: str                  # human-readable question
    yes_price: Optional[float]  # cost in cents to buy YES (1–99), or None if unavailable
    no_price: Optional[float]   # cost in cents to buy NO  (1–99), or None if unavailable


@dataclass
class MatchedPair:
    kalshi: FlatMarket
    poly: FlatMarket
    similarity: float
    # Arbitrage edges: positive = guaranteed profit per $1 invested, None = leg unavailable
    # Strategy A: buy YES on Kalshi + buy NO on Polymarket
    edge_yes_k_no_p: Optional[float]
    # Strategy B: buy NO on Kalshi + buy YES on Polymarket
    edge_no_k_yes_p: Optional[float]

    @property
    def best_edge(self) -> float:
        edges = [e for e in (self.edge_yes_k_no_p, self.edge_no_k_yes_p) if e is not None]
        return max(edges) if edges else float("-inf")

    @property
    def best_strategy(self) -> str:
        if self.edge_yes_k_no_p is not None and (
            self.edge_no_k_yes_p is None or self.edge_yes_k_no_p >= self.edge_no_k_yes_p
        ):
            return f"YES on Kalshi @ {self.kalshi.yes_price:.1f}¢  +  NO on Polymarket @ {self.poly.no_price:.1f}¢"
        return f"NO on Kalshi @ {self.kalshi.no_price:.1f}¢  +  YES on Polymarket @ {self.poly.yes_price:.1f}¢"


# ---------------------------------------------------------------------------
# Market fetchers
# ---------------------------------------------------------------------------

def fetch_kalshi_markets() -> list[FlatMarket]:
    from src.indexers.kalshi.client import KalshiClient

    print("Fetching Kalshi markets...", flush=True)
    client = KalshiClient()
    markets: list[FlatMarket] = []
    try:
        for batch, _cursor in client.iter_events(limit=200):
            for m in batch:
                yes = _valid_price(m.yes_ask)
                no  = _valid_price(m.no_ask)
                # Skip entirely if neither side has a real price
                if yes is None and no is None:
                    continue
                sub = m.yes_sub_title or ""
                base = m.title or m.ticker
                label = f"{base} — {sub}" if sub and sub.lower() != base.lower() else base
                markets.append(FlatMarket(
                    platform="kalshi",
                    market_id=m.ticker,
                    label=label,
                    yes_price=yes,
                    no_price=no,
                ))
    finally:
        client.close()
    print(f"  {len(markets)} Kalshi markets loaded", flush=True)
    return markets


def fetch_polymarket_markets() -> list[FlatMarket]:
    from src.indexers.polymarket.client import PolymarketClient

    print("Fetching Polymarket markets...", flush=True)
    client = PolymarketClient()
    markets: list[FlatMarket] = []
    try:
        for batch, _offset in client.iter_markets(limit=500, active=True, closed=False):
            for m in batch:
                try:
                    outcomes = json.loads(m.outcomes) if isinstance(m.outcomes, str) else m.outcomes
                    prices   = json.loads(m.outcome_prices) if isinstance(m.outcome_prices, str) else m.outcome_prices
                    # Only handle binary (Yes/No) markets for now
                    if len(outcomes) != 2:
                        continue
                    p = [float(x) * 100 for x in prices]
                    yes_price = p[0] if 1 <= p[0] <= 99 else None
                    no_price  = p[1] if 1 <= p[1] <= 99 else None
                    if yes_price is None and no_price is None:
                        continue
                    markets.append(FlatMarket(
                        platform="polymarket",
                        market_id=m.condition_id or m.id,
                        label=m.question or m.slug,
                        yes_price=yes_price,
                        no_price=no_price,
                    ))
                except (json.JSONDecodeError, ValueError, TypeError, IndexError):
                    continue
    finally:
        client.close()
    print(f"  {len(markets)} Polymarket markets loaded", flush=True)
    return markets


# ---------------------------------------------------------------------------
# Compatibility validation  (filters out "same topic, different conditions")
# ---------------------------------------------------------------------------

import re as _re

def _extract_numbers(text: str) -> list[float]:
    """Pull every numeric value out of a label (ignores years 2020–2030)."""
    nums = []
    for m in _re.finditer(r'\b(\d+(?:\.\d+)?)\b', text.replace(",", "")):
        v = float(m.group(1))
        if 2020 <= v <= 2030:          # skip years — they're contextual, not thresholds
            continue
        nums.append(v)
    return nums


def _extract_constraints(text: str) -> dict:
    """Return structured numeric constraints from a label."""
    t = text.lower().replace(",", "")
    c: dict = {"above": [], "below": [], "exactly": [], "between": [], "dollars": [], "pct": []}

    for m in _re.finditer(r'(?:above|more than|over|at least|≥)\s+\$?(\d+(?:\.\d+)?)', t):
        c["above"].append(float(m.group(1)))
    for m in _re.finditer(r'(?:below|less than|under|at most|≤|not more than)\s+\$?(\d+(?:\.\d+)?)', t):
        c["below"].append(float(m.group(1)))
    for m in _re.finditer(r'exactly\s+\$?(\d+(?:\.\d+)?)', t):
        c["exactly"].append(float(m.group(1)))
    for m in _re.finditer(r'between\s+\$?(\d+(?:\.\d+)?)\s*%?\s+and\s+\$?(\d+(?:\.\d+)?)', t):
        c["between"].append((float(m.group(1)), float(m.group(2))))
    for m in _re.finditer(r'\$(\d+(?:\.\d+)?)', t):
        c["dollars"].append(float(m.group(1)))
    for m in _re.finditer(r'(\d+(?:\.\d+)?)\s*%', t):
        v = float(m.group(1))
        if v not in (2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028):
            c["pct"].append(v)
    return c


def _ranges_overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
    return lo1 <= hi2 and lo2 <= hi1


def _is_compatible(k_label: str, p_label: str) -> bool:
    """
    Return False when the two labels are clearly about different conditions
    (different numeric thresholds, non-overlapping ranges, incompatible scopes).
    Returns True when we cannot determine incompatibility — err on the side of inclusion.
    """
    kc = _extract_constraints(k_label)
    pc = _extract_constraints(p_label)

    # --- Dollar threshold mismatch ---
    # e.g. Dogecoin "$0.05 or above" vs "$0.20" → ratio > 2× → incompatible
    if kc["dollars"] and pc["dollars"]:
        any_close = any(
            max(a, b) / max(min(a, b), 1e-9) < 3.0
            for a in kc["dollars"] for b in pc["dollars"]
        )
        if not any_close:
            return False

    # --- "above X" vs "exactly Y" where Y ≤ X → can never both be true ---
    for ka in kc["above"]:
        for pe in pc["exactly"]:
            if pe <= ka:
                return False
    for pa in pc["above"]:
        for ke in kc["exactly"]:
            if ke <= pa:
                return False

    # --- "exactly X" vs "exactly Y" where X ≠ Y ---
    if kc["exactly"] and pc["exactly"]:
        if not any(abs(a - b) < 0.01 for a in kc["exactly"] for b in pc["exactly"]):
            return False

    # --- Non-overlapping "between" ranges ---
    for k_lo, k_hi in kc["between"]:
        for p_lo, p_hi in pc["between"]:
            if not _ranges_overlap(k_lo, k_hi, p_lo, p_hi):
                return False

    # --- "between X and Y" (Kalshi) vs "above Z" (Poly) where Z > Y → incompatible ---
    for k_lo, k_hi in kc["between"]:
        for pa in pc["above"]:
            if pa > k_hi:
                return False
    for p_lo, p_hi in pc["between"]:
        for ka in kc["above"]:
            if ka > p_hi:
                return False

    # --- "above X" vs "between Y and Z": if Y > X the "above" is almost always TRUE
    #     while the "between" is a narrow specific bracket — different questions.
    # e.g. "above 1,800,000" vs "between 2,400,000 and 2,600,000"
    for ka in kc["above"]:
        for p_lo, p_hi in pc["between"]:
            if p_lo > ka * 1.1:   # entire "between" range sits well above the "above" threshold
                return False
    for pa in pc["above"]:
        for k_lo, k_hi in kc["between"]:
            if k_lo > pa * 1.1:
                return False

    # --- Non-overlapping "between" ranges (strict — adjacent ranges like 0-3 / 3-6 also differ) ---
    # e.g. "between 3% and 6%" vs "between 0% and 3%"
    for k_lo, k_hi in kc["between"]:
        for p_lo, p_hi in pc["between"]:
            if not (k_lo < p_hi and p_lo < k_hi):   # strict overlap
                return False

    # --- Percentage mismatch: both have pct values, none close ---
    if kc["pct"] and pc["pct"]:
        any_close = any(
            max(a, b) / max(min(a, b), 1e-9) < 2.5
            for a in kc["pct"] for b in pc["pct"]
        )
        if not any_close:
            return False

    # --- Scope word conflicts ---
    scope_groups = [
        {"win", "wins", "winner"},                        # winning an event
        {"participate", "participation", "participates"},  # just taking part
        {"top 6", "top 4", "top 3", "top 2"},            # finishing range
        {"1st", "2nd", "3rd", "4th", "5th", "6th",
         "1st place", "2nd place", "3rd place"},          # exact finish
        {"this week", "this month"},                       # short window
        {"season", "year", "full season"},                 # long window
        {"voted off", "eliminated"},                       # reality-tv specific
    ]
    kl, pl = k_label.lower(), p_label.lower()
    for group in scope_groups:
        in_k = {w for w in group if w in kl}
        in_p = {w for w in group if w in pl}
        # Both labels reference the group but with different members → flag
        if in_k and in_p and in_k.isdisjoint(in_p):
            # Exception: "win" and "winner" are the same concept → allow
            pass  # fine-grained conflicts handled below

    # "this week" / short-term vs season-long or year-long scope
    short = {"this week", "this month", "today"}
    long_ = {"full season", "2025-26", "the year"}
    k_short = any(w in kl for w in short)
    p_short = any(w in pl for w in short)
    k_long  = any(w in kl for w in long_)
    p_long  = any(w in pl for w in long_)
    if (k_short and p_long) or (p_short and k_long):
        return False

    # "before <specific date>" vs "in <year>" — narrow window vs whole-year scope
    # e.g. "before Mar 14 2026" vs "in 2026" — clearly different time frames
    k_before_date = bool(_re.search(r'\bbefore\s+\w+\s+\d+', kl))
    p_before_date = bool(_re.search(r'\bbefore\s+\w+\s+\d+', pl))
    k_in_year = bool(_re.search(r'\bin\s+20\d\d\b', kl))
    p_in_year = bool(_re.search(r'\bin\s+20\d\d\b', pl))
    if (k_before_date and p_in_year and not p_before_date) or \
       (p_before_date and k_in_year and not k_before_date):
        return False

    # "participate" vs "win"
    k_participate = any(w in kl for w in ("participat",))
    p_win         = any(w in pl for w in ("win ", "wins ", "winner"))
    p_participate = any(w in pl for w in ("participat",))
    k_win         = any(w in kl for w in ("win ", "wins ", "winner"))
    if (k_participate and p_win) or (p_participate and k_win):
        return False

    # "voted off" / "eliminated" vs "win"
    k_elim = any(w in kl for w in ("voted off", "eliminated", "be voted"))
    p_elim = any(w in pl for w in ("voted off", "eliminated", "be voted"))
    if (k_elim and not p_elim and (k_win or p_win)):
        return False
    if (p_elim and not k_elim and (k_win or p_win)):
        return False

    # "top N" vs "Nth place" (finish in top 6 ≠ finish in 3rd)
    k_top = _re.search(r'top\s+(\d+)', kl)
    p_place = _re.search(r'(\d+)(?:st|nd|rd|th)\s+place', pl)
    if k_top and p_place:
        return False
    p_top = _re.search(r'top\s+(\d+)', pl)
    k_place = _re.search(r'(\d+)(?:st|nd|rd|th)\s+place', kl)
    if p_top and k_place:
        return False

    return True


# ---------------------------------------------------------------------------
# Embedding + matching
# ---------------------------------------------------------------------------

def embed(texts: list[str], model) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=64)


def find_matches(
    kalshi: list[FlatMarket],
    poly: list[FlatMarket],
    threshold: float = 0.82,
) -> list[MatchedPair]:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    hf_token = os.getenv("HF_TOKEN")

    print("\nLoading Sentence-BERT model (all-MiniLM-L6-v2)...", flush=True)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", token=hf_token)

    print("Embedding Kalshi labels...", flush=True)
    k_labels = [m.label for m in kalshi]
    k_emb = embed(k_labels, model)

    print("Embedding Polymarket labels...", flush=True)
    p_labels = [m.label for m in poly]
    p_emb = embed(p_labels, model)

    print("Computing cosine similarity matrix...", flush=True)
    sim_matrix = cosine_similarity(k_emb, p_emb)  # (K, P)

    pairs: list[MatchedPair] = []
    for ki, km in enumerate(kalshi):
        # Best matching poly market for this kalshi market
        best_pi = int(np.argmax(sim_matrix[ki]))
        score = float(sim_matrix[ki, best_pi])
        if score < threshold:
            continue
        pm = poly[best_pi]

        # Reject pairs where numeric thresholds / scopes are clearly mismatched
        if not _is_compatible(km.label, pm.label):
            continue

        # Arbitrage edge = 100 - (cost of both legs)
        # Positive means guaranteed profit if both sides resolve on same event
        # Only compute when BOTH legs of a strategy are actually available
        edge_a = (100.0 - (km.yes_price + pm.no_price)
                  if km.yes_price is not None and pm.no_price is not None else None)
        edge_b = (100.0 - (km.no_price + pm.yes_price)
                  if km.no_price is not None and pm.yes_price is not None else None)

        # Skip pairs where no strategy is computable at all
        if edge_a is None and edge_b is None:
            continue

        pairs.append(MatchedPair(
            kalshi=km,
            poly=pm,
            similarity=score,
            edge_yes_k_no_p=edge_a,
            edge_no_k_yes_p=edge_b,
        ))

    # Sort: arb opportunities first, then by best edge, then similarity
    pairs.sort(key=lambda p: (p.best_edge > 0, p.best_edge, p.similarity), reverse=True)

    return pairs


# ---------------------------------------------------------------------------
# Display + storage
# ---------------------------------------------------------------------------

def print_matches(pairs: list[MatchedPair], top: int | None = None) -> None:
    arb_pairs   = [p for p in pairs if p.best_edge > 0]
    other_pairs = [p for p in pairs if p.best_edge <= 0]

    shown = pairs if top is None else pairs[:top]

    if arb_pairs:
        print(f"\n{'='*100}")
        print(f"  ARBITRAGE OPPORTUNITIES  ({len(arb_pairs)} found)")
        print(f"{'='*100}")
        for p in [x for x in shown if x.best_edge > 0]:
            _print_pair(p, highlight=True)
    else:
        print("\n  No pure arbitrage opportunities found at current prices.")

    remaining = [x for x in shown if x.best_edge <= 0]
    if remaining:
        print(f"\n{'='*100}")
        print(f"  CLOSE MATCHES  (similarity ≥ threshold, no arb edge)")
        print(f"{'='*100}")
        for p in remaining:
            _print_pair(p, highlight=False)


def _fmt_price(val: Optional[float]) -> str:
    return f"{val:.1f}¢" if val is not None else "N/A"


def _print_pair(p: MatchedPair, highlight: bool) -> None:
    tag = "  *** ARB ***" if highlight else ""
    print(f"\n  Similarity: {p.similarity:.3f}{tag}")
    print(f"  Kalshi    : {p.kalshi.label[:80]}")
    print(f"             Yes: {_fmt_price(p.kalshi.yes_price)}  |  No: {_fmt_price(p.kalshi.no_price)}")
    print(f"  Polymarket: {p.poly.label[:80]}")
    print(f"             Yes: {_fmt_price(p.poly.yes_price)}  |  No: {_fmt_price(p.poly.no_price)}")
    if highlight:
        print(f"  Strategy  : {p.best_strategy}")
        print(f"  Edge      : +{p.best_edge:.1f}¢ per $1 wagered on each leg")
    else:
        best = p.best_edge
        print(f"  Best edge : {best:+.1f}¢  (need {abs(best):.1f}¢ more price movement for arb)")


def save_results(pairs: list[MatchedPair]) -> None:
    import csv

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Full matches as JSON
    json_path = DATA_DIR / f"matches_{ts}.json"
    records = []
    for p in pairs:
        records.append({
            "similarity": p.similarity,
            "best_edge_cents": p.best_edge,
            "best_strategy": p.best_strategy if p.best_edge > 0 else None,
            "kalshi": asdict(p.kalshi),
            "polymarket": asdict(p.poly),
        })
    json_path.write_text(json.dumps(records, indent=2))
    print(f"\nAll matches saved  → {json_path}")

    # Arbitrage opportunities as CSV
    arb = [p for p in pairs if p.best_edge > 0]
    if arb:
        csv_path = DATA_DIR / f"arb_{ts}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "similarity", "edge_cents", "strategy",
                "kalshi_id", "kalshi_label", "kalshi_yes", "kalshi_no",
                "poly_id", "poly_label", "poly_yes", "poly_no",
            ])
            for p in arb:
                writer.writerow([
                    f"{p.similarity:.4f}", f"{p.best_edge:.2f}", p.best_strategy,
                    p.kalshi.market_id, p.kalshi.label, p.kalshi.yes_price, p.kalshi.no_price,
                    p.poly.market_id, p.poly.label, p.poly.yes_price, p.poly.no_price,
                ])
        print(f"Arb opportunities  → {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]

    threshold = 0.82
    top: int | None = None

    for i, arg in enumerate(args):
        if arg == "--similarity" and i + 1 < len(args):
            threshold = float(args[i + 1])
        if arg == "--top" and i + 1 < len(args):
            top = int(args[i + 1])

    print(f"Similarity threshold: {threshold}")

    kalshi_markets = fetch_kalshi_markets()
    poly_markets   = fetch_polymarket_markets()

    if not kalshi_markets or not poly_markets:
        print("Could not fetch markets from one or both platforms.")
        sys.exit(1)

    pairs = find_matches(kalshi_markets, poly_markets, threshold=threshold)

    print(f"\n{len(pairs)} matched pairs above threshold {threshold}")

    print_matches(pairs, top=top)
    save_results(pairs)


if __name__ == "__main__":
    main()
