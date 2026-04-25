# Prediction Market Analysis

For clarification the code used to pull the data from the prediciton markets was open source and this was approved by the proffesor via email.

## Setup

Requires Python 3.9+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Quick Start

```bash
# 1) Pull fresh cross-platform matches and arb CSV
uv run arb_finder.py --similarity 0.82 --top 100

# 2) Rank top opportunities by ROI
uv run top_arb.py --top 20 --min-similarity 0.90

# 3) Open local UI to view/refresh top arbs
uv run streamlit run app.py
```

## Script Runbook

### 1) Fetch and match markets (Kalshi + Polymarket)

```bash
uv run arb_finder.py [--similarity 0.82] [--top 50]
```

Outputs:

- `data/arb/matches_<timestamp>.json`
- `data/arb/arb_<timestamp>.csv`

### 2) View current market lines only

```bash
uv run list_markets.py [--kalshi] [--polymarket]
```

Outputs:

- `data/markets/kalshi_<timestamp>.csv`
- `data/markets/polymarket_<timestamp>.csv`

### 3) Rank best arbitrage opportunities

```bash
uv run top_arb.py [--top 20] [--min-similarity 0.90] [--file data/arb/arb_*.csv]
```

Optional sizing arguments:

```bash
uv run top_arb.py --top 20 --bankroll 1000 --allocation-pct 0.02
```

Output:

- `data/top_arb/top_arb_<timestamp>.csv`

### 4) Create labeling dataset

```bash
# Single file
uv run label_pairs.py --n 500 --out data/labels/to_label.csv

# Train/val/test labeling split
uv run label_pairs.py --n 900 --split --split-prefix to_label --group-col kalshi_id
```

Outputs:

- `data/labels/to_label.csv` or split files:
  - `data/labels/to_label_train.csv`
  - `data/labels/to_label_val.csv`

### 5) Train classifier

```bash
# Basic training
uv run -m src.ml.train --labels data/labels/to_label.csv

# With explicit holdouts
uv run -m src.ml.train \
  --labels data/labels/to_label_train.csv \
  --val-labels data/labels/to_label_val.csv
```

Outputs (default `models/`):

- `pair_classifier.pkl`
- `cv_report.json`
- optionally `val_report.json`

### 6) Score new arb CSV with trained model

```bash
uv run -m src.ml.score --model models/pair_classifier.pkl --data data/arb/arb_LATEST.csv
```

Useful filter mode:

```bash
uv run -m src.ml.score \
  --model models/pair_classifier.pkl \
  --data data/arb/arb_LATEST.csv \
  --true-only --min-true-prob 0.90
```

Output:

- `<input_stem>_scored.csv` (or `--out` path)

### 7) Local frontend

```bash
uv run streamlit run app.py
```

Use the button to refresh from live markets and rebuild top-arb outputs.

## Main CLI (menu style)

```bash
uv run main.py analyze [analysis_name|all]
uv run main.py index
uv run main.py package
```

Equivalent make targets:

```bash
make analyze
make index
make package
```

## Repo Layout

```text
app.py                 # Streamlit frontend
arb_finder.py          # Live market fetch + match + arb output
list_markets.py        # Print and save current market lines
label_pairs.py         # Build labeling CSVs
top_arb.py             # Rank opportunities by ROI
src/ml/train.py        # Train classifier
src/ml/score.py        # Score and filter pairs
data/                  # Generated CSV/JSON/Parquet files
models/                # Trained model + evaluation reports
```

