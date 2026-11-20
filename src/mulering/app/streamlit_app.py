"""Minimal Streamlit stub: ring queue + ring view. Expand per docs/03_APP_FLOW.md."""
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Mule Ring Detector", layout="wide")
st.title("Money Mule Ring Detector")
st.caption("Synthetic data. Research prototype, not a production AML system.")

path = Path("reports/rings.parquet")
if not path.exists():
    st.info("No rings found yet. Run: make data ingest features train-baseline rings")
    st.stop()

rings = pd.read_parquet(path)
min_score = st.slider("Minimum ring score", 0.0, 1.0, 0.5, 0.01)
view = rings[rings["ring_score"] >= min_score].sort_values("ring_score", ascending=False)
st.dataframe(view[["pred_ring_id", "ring_score", "pattern_guess", "n_accounts", "total_amount"]],
             use_container_width=True)
