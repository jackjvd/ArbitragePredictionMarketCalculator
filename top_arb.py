"""
Show the top N arb opportunities ranked by ROI from the latest arb CSV.

Usage:
    uv run top_arb.py [--top 20] [--min-similarity 0.90] [--file path/to/arb.csv]

ROI = edge / total_cost * 100%
   where total_cost = cost of both bet legs combined (= 100 - edge_cents)
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path


ARB_DIR  = Path("data/arb")
OUT_DIR  = Path("data/top_arb")


def latest_arb_file() -> Path:
    files = sorted(ARB_DIR.glob("arb_*.csv"), reverse=True)
    if not files:
        print(f"No arb CSV files found in {ARB_DIR}/")
        sys.exit(1)
    return files[0]


def load_arb(path: Path, min_similarity: float) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                edge = float(row["edge_cents"])
                sim  = float(row["similarity"])
                k_yes = row["kalshi_yes"]
                k_no  = row["kalshi_no"]
                p_yes = row["poly_yes"]
                p_no  = row["poly_no"]

                # Require all four prices to be real (no N/A on any side)
                def _real(v: str) -> float | None:
                    try:
                        f = float(v)
                        return f if 1 <= f <= 99 else None
                    except (ValueError, TypeError):
                        return None

                k_yes_f = _real(k_yes)
                k_no_f  = _real(k_no)
                p_yes_f = _real(p_yes)
                p_no_f  = _real(p_no)

                if None in (k_yes_f, k_no_f, p_yes_f, p_no_f):
                    continue

                strategy = row["strategy"]
                if "YES on Kalshi" in strategy:
                    leg1, leg2 = k_yes_f, p_no_f
                else:
                    leg1, leg2 = k_no_f, p_yes_f
                if sim < min_similarity:
                    continue
                if edge <= 0:
                    continue

                total_cost = leg1 + leg2
                roi = (edge / total_cost) * 100.0

                rows.append({
                    "roi": roi,
                    "edge": edge,
                    "total_cost": total_cost,
                    "leg1": leg1,
                    "leg2": leg2,
                    "similarity": sim,
                    "strategy": strategy,
                    "kalshi_label": row["kalshi_label"],
                    "poly_label": row["poly_label"],
                    "kalshi_yes": k_yes,
                    "kalshi_no": k_no,
                    "poly_yes": p_yes,
                    "poly_no": p_no,
                })
            except (ValueError, KeyError):
                continue

    rows.sort(key=lambda r: r["roi"], reverse=True)
    return rows


def _fmt(val: str) -> str:
    try:
        f = float(val)
        return "N/A" if f >= 100 else f"{f:.1f}¢"
    except (ValueError, TypeError):
        return "N/A"


def save_top(rows: list[dict], top: int, source_file: Path) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"top_arb_{ts}.csv"
    shown = rows[:top]
    if shown:
        fieldnames = ["rank", "roi_pct", "edge_cents", "total_cost", "similarity",
                      "strategy", "kalshi_label", "kalshi_yes", "kalshi_no",
                      "poly_label", "poly_yes", "poly_no", "source_file"]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, r in enumerate(shown, 1):
                writer.writerow({
                    "rank": i,
                    "roi_pct": round(r["roi"], 2),
                    "edge_cents": round(r["edge"], 2),
                    "total_cost": round(r["total_cost"], 2),
                    "similarity": round(r["similarity"], 4),
                    "strategy": r["strategy"],
                    "kalshi_label": r["kalshi_label"],
                    "kalshi_yes": r["kalshi_yes"],
                    "kalshi_no": r["kalshi_no"],
                    "poly_label": r["poly_label"],
                    "poly_yes": r["poly_yes"],
                    "poly_no": r["poly_no"],
                    "source_file": source_file.name,
                })
    return path


def print_top(rows: list[dict], top: int) -> None:
    shown = rows[:top]
    if not shown:
        print("No qualifying arbitrage opportunities found.")
        return

    print(f"\n{'='*100}")
    print(f"  TOP {len(shown)} ARB OPPORTUNITIES BY ROI  (from {len(rows)} total)")
    print(f"{'='*100}\n")

    for i, r in enumerate(shown, 1):
        print(f"  #{i}  ROI: {r['roi']:.1f}%   Edge: +{r['edge']:.1f}¢   Cost: {r['total_cost']:.1f}¢   Similarity: {r['similarity']:.3f}")
        print(f"       Kalshi    : {r['kalshi_label'][:80]}")
        print(f"                   Yes: {_fmt(r['kalshi_yes'])}  |  No: {_fmt(r['kalshi_no'])}")
        print(f"       Polymarket: {r['poly_label'][:80]}")
        print(f"                   Yes: {_fmt(r['poly_yes'])}  |  No: {_fmt(r['poly_no'])}")
        print(f"       Strategy  : {r['strategy']}")
        print()


def main() -> None:
    args = sys.argv[1:]

    top = 20
    min_sim = 0.90
    file_path: Path | None = None

    i = 0
    while i < len(args):
        if args[i] == "--top" and i + 1 < len(args):
            top = int(args[i + 1]); i += 2
        elif args[i] == "--min-similarity" and i + 1 < len(args):
            min_sim = float(args[i + 1]); i += 2
        elif args[i] == "--file" and i + 1 < len(args):
            file_path = Path(args[i + 1]); i += 2
        else:
            i += 1

    path = file_path or latest_arb_file()
    print(f"Reading: {path}")
    print(f"Filters: min similarity={min_sim}, top={top}")

    rows = load_arb(path, min_similarity=min_sim)
    print_top(rows, top)

    if rows:
        print(f"  Best ROI:  {rows[0]['roi']:.1f}%")
        print(f"  Worst ROI (of shown): {rows[min(top, len(rows))-1]['roi']:.1f}%")
        print(f"  Total qualifying arbs: {len(rows)}")
        out = save_top(rows, top, path)
        print(f"  Saved → {out}\n")


if __name__ == "__main__":
    main()
