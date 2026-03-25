"""
Print every market and the prices you can bet at right now on Kalshi and Polymarket.
Also saves a timestamped CSV to data/markets/.

Usage:
    uv run list_markets.py [--kalshi] [--polymarket]

If no platform flag is given, both are shown.
Prices are in cents — e.g. 62¢ means you pay $0.62 to win $1.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("data/markets")


def _is_real_price(val: int | None) -> bool:
    """True if this is an actual bettable price (not a parlay placeholder)."""
    return val is not None and 1 <= val <= 99


def _save_csv(rows: list[dict], platform: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = DATA_DIR / f"{platform}_{ts}.csv"
    if rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return path


def print_kalshi_markets() -> None:
    from src.indexers.kalshi.client import KalshiClient

    print("=" * 100)
    print("KALSHI  (cents per $1 payout)")
    print("=" * 100)
    print(f"{'Market':<55}  Bet prices")
    print("-" * 100)

    client = KalshiClient()
    rows: list[dict] = []
    try:
        for markets, _cursor in client.iter_events(limit=200):
            for m in markets:
                if not (_is_real_price(m.yes_ask) or _is_real_price(m.no_ask)):
                    continue
                sub = m.yes_sub_title or ""
                base = m.title or m.ticker
                label = f"{base} — {sub}" if sub and sub.lower() != base.lower() else base

                yes_str = f"{m.yes_ask}.0¢" if _is_real_price(m.yes_ask) else "N/A"
                no_str  = f"{m.no_ask}.0¢"  if _is_real_price(m.no_ask)  else "N/A"
                price_str = "  |  ".join(p for p in [
                    f"Yes: {yes_str}" if _is_real_price(m.yes_ask) else "",
                    f"No: {no_str}"   if _is_real_price(m.no_ask)  else "",
                ] if p)

                print(f"{label[:54]:<55}  {price_str}")
                rows.append({
                    "platform": "kalshi",
                    "ticker": m.ticker,
                    "label": label,
                    "yes_price": m.yes_ask if _is_real_price(m.yes_ask) else "",
                    "no_price":  m.no_ask  if _is_real_price(m.no_ask)  else "",
                    "status": m.status,
                })
    finally:
        client.close()

    path = _save_csv(rows, "kalshi")
    print("-" * 100)
    print(f"{len(rows)} Kalshi markets with live prices")
    print(f"Saved → {path}\n")


def print_polymarket_markets() -> None:
    from src.indexers.polymarket.client import PolymarketClient

    print("=" * 100)
    print("POLYMARKET  (cents per $1 payout)")
    print("=" * 100)
    print(f"{'Market':<55}  Bet prices")
    print("-" * 100)

    client = PolymarketClient()
    rows: list[dict] = []
    try:
        for markets, _offset in client.iter_markets(limit=500, active=True, closed=False):
            for m in markets:
                question = (m.question or m.slug)[:54]

                try:
                    outcomes = json.loads(m.outcomes) if isinstance(m.outcomes, str) else m.outcomes
                    prices   = json.loads(m.outcome_prices) if isinstance(m.outcome_prices, str) else m.outcome_prices
                    price_str = "  |  ".join(
                        f"{o}: {float(p) * 100:.1f}¢"
                        for o, p in zip(outcomes, prices)
                    )
                    price_map = {o: round(float(p) * 100, 4) for o, p in zip(outcomes, prices)}
                except (json.JSONDecodeError, ValueError, TypeError):
                    price_str = str(m.outcome_prices)
                    price_map = {}

                print(f"{question:<55}  {price_str}")
                rows.append({
                    "platform": "polymarket",
                    "condition_id": m.condition_id,
                    "question": m.question or m.slug,
                    "outcomes": "|".join(str(o) for o in (price_map.keys() if price_map else [])),
                    "prices_cents": "|".join(str(v) for v in (price_map.values() if price_map else [])),
                    "active": m.active,
                    "closed": m.closed,
                })
    finally:
        client.close()

    path = _save_csv(rows, "polymarket")
    print("-" * 100)
    print(f"{len(rows)} active Polymarket markets")
    print(f"Saved → {path}\n")


def main() -> None:
    args = sys.argv[1:]
    show_kalshi     = "--kalshi"     in args or not any(a in args for a in ("--kalshi", "--polymarket"))
    show_polymarket = "--polymarket" in args or not any(a in args for a in ("--kalshi", "--polymarket"))

    if show_kalshi:
        print_kalshi_markets()

    if show_polymarket:
        print_polymarket_markets()


if __name__ == "__main__":
    main()
