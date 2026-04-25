from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(".")
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"


def list_files(pattern: str, base_dir: Path | None = None) -> list[Path]:
    base = base_dir or ROOT
    return sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


def latest_file(pattern: str, base_dir: Path | None = None) -> Path | None:
    files = list_files(pattern, base_dir)
    return files[0] if files else None


def safe_read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def safe_read_table(path: Path, max_rows: int = 5000) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path).head(max_rows)
    if suffix == ".json":
        data = safe_read_json(path)
        if data is None:
            return pd.DataFrame()
        if isinstance(data, list):
            return pd.DataFrame(data).head(max_rows)
        if isinstance(data, dict):
            return pd.DataFrame([data]).head(max_rows)
    if suffix == ".parquet":
        return pd.read_parquet(path).head(max_rows)
    return pd.DataFrame()


def project_snapshot() -> dict[str, Any]:
    arb_csv = list_files("arb/*.csv", DATA_DIR)
    label_csv = list_files("labels/*.csv", DATA_DIR)
    top_arb_csv = list_files("top_arb/*.csv", DATA_DIR)
    model_json = list_files("*.json", MODELS_DIR)
    model_pkl = list_files("*.pkl", MODELS_DIR)

    return {
        "arb_csv_count": len(arb_csv),
        "labels_count": len(label_csv),
        "top_arb_count": len(top_arb_csv),
        "model_json_count": len(model_json),
        "model_pkl_count": len(model_pkl),
        "latest_arb_csv": str(arb_csv[0]) if arb_csv else None,
        "latest_label_csv": str(label_csv[0]) if label_csv else None,
        "latest_top_arb_csv": str(top_arb_csv[0]) if top_arb_csv else None,
    }


def label_quality_stats(df: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    if df.empty:
        return stats

    if "label" in df.columns:
        labels = df["label"].astype(str).str.strip()
        unresolved = labels.isin({"", "nan", "-1"})
        invalid = ~labels.isin({"0", "1", "2", "", "nan", "-1"})
        stats["rows"] = len(df)
        stats["unresolved"] = int(unresolved.sum())
        stats["invalid"] = int(invalid.sum())
        stats["completion_pct"] = round((1.0 - unresolved.mean()) * 100.0, 2)
    if {"kalshi_id", "poly_id"}.issubset(df.columns):
        dupes = df.duplicated(subset=["kalshi_id", "poly_id"]).sum()
        stats["duplicate_pairs"] = int(dupes)
    if {"label_hint", "label"}.issubset(df.columns):
        label = df["label"].astype(str).str.strip()
        hint = df["label_hint"].astype(str).str.strip()
        mask = label.isin({"0", "1", "2"}) & hint.isin({"0", "1", "2"})
        if int(mask.sum()) > 0:
            stats["hint_agreement_pct"] = round(float((label[mask] == hint[mask]).mean() * 100.0), 2)
    return stats


def report_to_frame(report: dict[str, Any] | None) -> pd.DataFrame:
    if not report:
        return pd.DataFrame()
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {"label": label}
            row.update(values)
            rows.append(row)
    return pd.DataFrame(rows)

