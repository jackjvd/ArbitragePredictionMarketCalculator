from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.ui.data_loader import (
    DATA_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    label_quality_stats,
    latest_file,
    list_files,
    project_snapshot,
    report_to_frame,
    safe_read_json,
    safe_read_table,
)
from src.ui.runner import run_command_streaming


def _file_age_text(path: Path) -> str:
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    delta = now - modified
    mins = int(delta.total_seconds() // 60)
    if mins < 1:
        return "updated just now"
    if mins < 60:
        return f"updated {mins}m ago"
    hours = mins // 60
    if hours < 24:
        return f"updated {hours}h ago"
    days = hours // 24
    return f"updated {days}d ago"


def _bettable_price_mask(df: pd.DataFrame) -> pd.Series:
    required = ["kalshi_yes", "kalshi_no", "poly_yes", "poly_no"]
    if not set(required).issubset(df.columns):
        return pd.Series([True] * len(df), index=df.index)
    numeric = df[required].apply(pd.to_numeric, errors="coerce")
    return numeric.ge(1).all(axis=1) & numeric.le(99).all(axis=1)


def render_overview() -> None:
    st.subheader("Overview")
    snap = project_snapshot()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Arb CSVs", snap["arb_csv_count"])
    c2.metric("Label CSVs", snap["labels_count"])
    c3.metric("Top Arb CSVs", snap["top_arb_count"])
    c4.metric("Model JSONs", snap["model_json_count"])
    c5.metric("Model PKLs", snap["model_pkl_count"])

    st.caption("Latest key files")
    st.write(
        {
            "latest_arb_csv": snap["latest_arb_csv"],
            "latest_label_csv": snap["latest_label_csv"],
            "latest_top_arb_csv": snap["latest_top_arb_csv"],
        }
    )

    cv = safe_read_json(MODELS_DIR / "cv_report.json")
    val = safe_read_json(MODELS_DIR / "val_report.json")
    test = safe_read_json(MODELS_DIR / "test_report.json")
    if any([cv, val, test]):
        st.markdown("**Latest model reports**")
        cols = st.columns(3)
        for idx, (name, report) in enumerate([("CV", cv), ("Val", val), ("Test", test)]):
            if isinstance(report, dict) and "accuracy" in report:
                cols[idx].metric(f"{name} accuracy", f"{report['accuracy']:.3f}")
            else:
                cols[idx].metric(f"{name} accuracy", "N/A")


def render_arbitrage_explorer() -> None:
    st.subheader("Arbitrage Explorer")
    files = list_files("arb/arb_*.csv", DATA_DIR)
    if not files:
        st.warning("No arbitrage CSV files found in data/arb.")
        return

    selected = st.selectbox("Arb CSV", files, format_func=lambda p: p.name)
    df = pd.read_csv(selected)
    st.caption(f"{len(df)} rows loaded from {selected}")

    sim_range = st.slider("Similarity range", 0.0, 1.0, (0.82, 1.0), 0.001)
    min_edge = st.number_input("Minimum edge_cents", value=0.0, step=0.5)
    keyword = st.text_input("Keyword filter (labels or strategy)")

    filtered = df[(df["similarity"] >= sim_range[0]) & (df["similarity"] <= sim_range[1]) & (df["edge_cents"] >= min_edge)]
    if keyword:
        mask = (
            filtered["kalshi_label"].fillna("").str.contains(keyword, case=False)
            | filtered["poly_label"].fillna("").str.contains(keyword, case=False)
            | filtered["strategy"].fillna("").str.contains(keyword, case=False)
        )
        filtered = filtered[mask]

    st.metric("Filtered rows", len(filtered))
    st.dataframe(filtered, use_container_width=True, height=450)
    st.download_button(
        "Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected.stem}_filtered.csv",
        mime="text/csv",
    )

    json_files = list_files("arb/matches_*.json", DATA_DIR)
    if json_files:
        with st.expander("Matching JSON preview"):
            js_sel = st.selectbox("Matches JSON", json_files, format_func=lambda p: p.name)
            payload = safe_read_json(js_sel)
            if isinstance(payload, list):
                st.dataframe(pd.DataFrame(payload).head(100), use_container_width=True)
            else:
                st.json(payload)


def render_labeling_workspace() -> None:
    st.subheader("Labeling Workspace")
    files = list_files("labels/*.csv", DATA_DIR)
    if not files:
        st.warning("No label CSV files found in data/labels.")
        return

    selected = st.selectbox("Label file", files, format_func=lambda p: p.name)
    df = pd.read_csv(selected)
    stats = label_quality_stats(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", stats.get("rows", 0))
    c2.metric("Completion %", stats.get("completion_pct", 0))
    c3.metric("Unresolved", stats.get("unresolved", 0))
    c4.metric("Invalid labels", stats.get("invalid", 0))
    if "duplicate_pairs" in stats or "hint_agreement_pct" in stats:
        c5, c6 = st.columns(2)
        c5.metric("Duplicate pairs", stats.get("duplicate_pairs", 0))
        c6.metric("Hint agreement %", stats.get("hint_agreement_pct", 0))

    st.dataframe(df, use_container_width=True, height=450)
    st.download_button(
        "Download current label file",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=selected.name,
        mime="text/csv",
    )


def _render_report(report_path: Path, label: str) -> None:
    report = safe_read_json(report_path)
    st.markdown(f"**{label}**: `{report_path}`")
    if not isinstance(report, dict):
        st.info("Report missing or unreadable.")
        return

    if "accuracy" in report:
        st.metric(f"{label} accuracy", f"{report['accuracy']:.3f}")
    st.dataframe(report_to_frame(report), use_container_width=True)


def render_ml_panel() -> None:
    st.subheader("ML Panel")
    _render_report(MODELS_DIR / "cv_report.json", "Cross-validation report")
    _render_report(MODELS_DIR / "val_report.json", "Validation report")
    _render_report(MODELS_DIR / "test_report.json", "Test report")

    st.markdown("**Score a file with trained model**")
    arb_files = list_files("arb/arb_*.csv", DATA_DIR)
    model_default = MODELS_DIR / "pair_classifier.pkl"
    if not arb_files:
        st.info("No arb CSV files available to score.")
        return
    if not model_default.exists():
        st.info("No model found at models/pair_classifier.pkl")
        return

    input_file = st.selectbox("Input arb CSV", arb_files, format_func=lambda p: p.name, key="score_input_file")
    min_true_prob = st.slider("Min true-match probability", 0.0, 1.0, 0.9, 0.01)
    true_only = st.checkbox("Keep true matches only", value=True)
    if st.button("Run scorer", key="run_scorer_button"):
        cmd = [
            "uv",
            "run",
            "-m",
            "src.ml.score",
            "--model",
            str(model_default),
            "--data",
            str(input_file),
        ]
        if true_only:
            cmd.extend(["--true-only", "--min-true-prob", str(min_true_prob)])
        result = run_command_streaming(cmd)
        if result.exit_code == 0:
            st.success("Scoring complete.")
        else:
            st.error("Scoring failed.")


def render_top_arb() -> None:
    st.subheader("Top Arbitrage")

    st.markdown("### Refresh data")
    c1, c2, c3 = st.columns(3)
    refresh_similarity = c1.slider("Similarity threshold", 0.5, 1.0, 0.82, 0.01)
    refresh_top = c2.number_input("Top rows to compute", min_value=5, max_value=500, value=100, step=5)
    refresh_min_similarity = c3.slider("Top ranking min similarity", 0.5, 1.0, 0.90, 0.01)
    c4, c5 = st.columns(2)
    model_path = Path("models/pair_classifier.pkl")
    use_model_gate = c4.checkbox("Gate with classifier (pred_label=2)", value=model_path.exists())
    min_true_prob = c5.slider("Classifier min pred_prob_2", 0.5, 1.0, 0.90, 0.01)
    if use_model_gate and not model_path.exists():
        st.warning("Classifier gate requested, but models/pair_classifier.pkl was not found.")
        use_model_gate = False

    st.caption("Runs live fetch + match (`arb_finder.py`), then rebuilds top-arb ranking (`top_arb.py`).")
    if st.button("Refresh from latest Kalshi + Polymarket prices", type="primary"):
        quality_stats: dict[str, int] = {}
        with st.status("Refreshing data from market APIs...", expanded=True):
            arb_cmd = [
                "uv",
                "run",
                "arb_finder.py",
                "--similarity",
                str(refresh_similarity),
                "--top",
                str(int(refresh_top)),
            ]
            arb_result = run_command_streaming(arb_cmd)
            if arb_result.exit_code != 0:
                st.error("arb_finder failed. Check output below.")
                with st.expander("arb_finder output"):
                    st.code(arb_result.output, language="bash")
                return

            latest_arb = latest_file("arb/arb_*.csv", DATA_DIR)
            if latest_arb is None:
                st.error("arb_finder completed but no arb CSV was found.")
                return

            arb_df = pd.read_csv(latest_arb)
            quality_stats["raw_rows"] = len(arb_df)
            valid_price_mask = _bettable_price_mask(arb_df)
            quality_stats["dropped_missing_or_non_bettable_prices"] = int((~valid_price_mask).sum())
            filtered_for_pricing = arb_df[valid_price_mask].copy()
            quality_stats["rows_after_price_filter"] = len(filtered_for_pricing)

            input_for_top = latest_arb

            if use_model_gate:
                gated_path = latest_arb.parent / f"{latest_arb.stem}_gated_true_only.csv"
                score_cmd = [
                    "uv",
                    "run",
                    "-m",
                    "src.ml.score",
                    "--model",
                    str(model_path),
                    "--data",
                    str(input_for_top),
                    "--out",
                    str(gated_path),
                    "--true-only",
                    "--min-true-prob",
                    str(min_true_prob),
                ]
                score_result = run_command_streaming(score_cmd)
                if score_result.exit_code != 0:
                    st.error("Classifier gating failed. Check output below.")
                    with st.expander("score output"):
                        st.code(score_result.output, language="bash")
                    return
                input_for_top = gated_path
                gated_df = pd.read_csv(gated_path)
                quality_stats["rows_after_model_gate"] = len(gated_df)
                quality_stats["dropped_by_model_gate"] = max(
                    0,
                    quality_stats["rows_after_price_filter"] - quality_stats["rows_after_model_gate"],
                )

            top_cmd = [
                "uv",
                "run",
                "top_arb.py",
                "--top",
                str(int(refresh_top)),
                "--min-similarity",
                str(refresh_min_similarity),
                "--file",
                str(input_for_top),
            ]
            top_result = run_command_streaming(top_cmd)
            if top_result.exit_code != 0:
                st.error("top_arb failed. Check output below.")
                with st.expander("top_arb output"):
                    st.code(top_result.output, language="bash")
                return
        st.success("Refresh complete. Latest files generated.")
        stat_cols = st.columns(4)
        stat_cols[0].metric("Raw arb rows", quality_stats.get("raw_rows", 0))
        stat_cols[1].metric(
            "Dropped missing/non-bettable prices",
            quality_stats.get("dropped_missing_or_non_bettable_prices", 0),
        )
        stat_cols[2].metric("After price filter", quality_stats.get("rows_after_price_filter", 0))
        if use_model_gate:
            stat_cols[3].metric("After model gate", quality_stats.get("rows_after_model_gate", 0))
        else:
            stat_cols[3].metric("After model gate", "N/A")

    st.divider()
    st.markdown("### Display")
    csv_files = list_files("top_arb/top_arb_*.csv", DATA_DIR)
    md_files = list_files("top_arb/*_pretty.md", DATA_DIR)
    if not csv_files and not md_files:
        st.warning("No top_arb output files found in data/top_arb.")
        return

    if csv_files:
        csv_sel = st.selectbox("Top arb CSV", csv_files, format_func=lambda p: p.name)
        df = pd.read_csv(csv_sel)
        st.caption(f"{csv_sel.name} — {_file_age_text(csv_sel)}")
        if datetime.now(tz=timezone.utc).timestamp() - csv_sel.stat().st_mtime > 60 * 60:
            st.warning("This output is older than 1 hour. Click refresh above for near-live data.")

        valid_mask = _bettable_price_mask(df)
        dropped_in_display = int((~valid_mask).sum())
        if dropped_in_display > 0:
            st.warning(f"Excluded {dropped_in_display} row(s) with missing or non-bettable prices.")
            df = df[valid_mask].copy()

        if df.empty:
            st.info("No rows left after filters.")
            return

        max_rows = max(1, len(df))
        rows_to_show = st.slider("How many opportunities to display", min_value=1, max_value=max_rows, value=min(20, max_rows))

        # Ensure both market ticket labels are easy to compare.
        preferred_cols = [
            "rank",
            "roi_pct",
            "edge_cents",
            "similarity",
            "kalshi_label",
            "poly_label",
            "strategy",
            "source_file",
        ]
        display_cols = [c for c in preferred_cols if c in df.columns]
        table = df.head(rows_to_show)
        if display_cols:
            table = table[display_cols]

        st.metric("Showing opportunities", len(table))
        st.dataframe(table, use_container_width=True, height=500)
    if md_files:
        md_sel = st.selectbox("Pretty markdown", md_files, format_func=lambda p: p.name)
        st.markdown(md_sel.read_text())


def render_data_browser() -> None:
    st.subheader("Data Browser")
    roots = [DATA_DIR, MODELS_DIR, OUTPUT_DIR]
    root = st.selectbox("Root directory", roots, format_func=lambda p: str(p))
    files = [p for p in root.rglob("*") if p.is_file()]
    if not files:
        st.info("No files found.")
        return
    selected = st.selectbox("File", files, format_func=lambda p: str(p.relative_to(Path("."))))
    st.caption(f"{selected} ({selected.stat().st_size} bytes)")

    if selected.suffix.lower() in {".csv", ".json", ".parquet"}:
        df = safe_read_table(selected, max_rows=1000)
        if not df.empty:
            st.dataframe(df, use_container_width=True, height=450)
        else:
            payload = safe_read_json(selected)
            if payload is not None:
                st.json(payload)
            else:
                st.info("Preview unavailable for this file.")
    elif selected.suffix.lower() in {".md", ".txt", ".log"}:
        st.text(selected.read_text()[:100000])
    else:
        st.info("Preview not supported for this file type.")


def render_run_center() -> None:
    st.subheader("Run Center")
    st.caption("Only approved project commands are allowed.")
    action = st.selectbox(
        "Action",
        [
            "Run arb finder",
            "Generate label splits",
            "Train model",
            "Score arb csv",
            "Run top arb ranking",
            "Run analyze all",
        ],
    )

    command: list[str]
    if action == "Run arb finder":
        similarity = st.slider("Similarity threshold", 0.5, 1.0, 0.82, 0.01)
        top = st.number_input("Top rows to print", min_value=1, value=50)
        command = ["uv", "run", "arb_finder.py", "--similarity", str(similarity), "--top", str(int(top))]
    elif action == "Generate label splits":
        n = st.number_input("Sample rows", min_value=100, value=900, step=50)
        seed = st.number_input("Seed", min_value=0, value=42)
        command = [
            "uv",
            "run",
            "label_pairs.py",
            "--n",
            str(int(n)),
            "--split",
            "--split-prefix",
            "to_label",
            "--group-col",
            "kalshi_id",
            "--seed",
            str(int(seed)),
        ]
    elif action == "Train model":
        command = [
            "uv",
            "run",
            "-m",
            "src.ml.train",
            "--labels",
            "data/labels/to_label_train.csv",
            "--val-labels",
            "data/labels/to_label_val.csv",
        ]
    elif action == "Score arb csv":
        data_file = latest_file("arb/arb_*.csv", DATA_DIR)
        command = [
            "uv",
            "run",
            "-m",
            "src.ml.score",
            "--model",
            "models/pair_classifier.pkl",
            "--data",
            str(data_file) if data_file else "data/arb/arb_LATEST.csv",
            "--true-only",
            "--min-true-prob",
            "0.90",
        ]
    elif action == "Run top arb ranking":
        command = ["uv", "run", "top_arb.py", "--top", "20", "--min-similarity", "0.90"]
    else:
        command = ["uv", "run", "main.py", "analyze", "all"]

    st.code(" ".join(command), language="bash")
    st.warning("Long-running jobs can take several minutes. Watch output below.")
    if st.button("Execute action"):
        result = run_command_streaming(command)
        if result.exit_code == 0:
            st.success("Command completed successfully.")
            for line in result.output.splitlines():
                if "Saved" in line or "→" in line:
                    st.write(line)
        else:
            st.error("Command failed.")
        with st.expander("Full command output"):
            st.code(result.output, language="bash")


PAGE_RENDERERS: dict[str, Any] = {
    "Overview": render_overview,
    "Arbitrage Explorer": render_arbitrage_explorer,
    "Labeling Workspace": render_labeling_workspace,
    "ML Panel": render_ml_panel,
    "Top Arb View": render_top_arb,
    "Data Browser": render_data_browser,
    "Run Center": render_run_center,
}

