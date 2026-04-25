from __future__ import annotations

import streamlit as st

from src.ui.views import render_top_arb


def main() -> None:
    st.set_page_config(page_title="Top Arbitrage Frontend", layout="wide")
    st.title("Top Arbitrage Frontend")
    st.caption("Display top arbitrage outputs with adjustable result count.")
    render_top_arb()


if __name__ == "__main__":
    main()

