"""Interactive Streamlit Dashboard for Anacity Facility Usage Prediction System."""

import os
import json
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Anacity Facility Usage Prediction System",
    page_icon="🏢",
    layout="wide",
)

st.title("🏢 Anacity Facility Usage Prediction System")
st.markdown(
    "**Anacity (Part of Anarock Group) AI Engineer Assignment** — Intelligent facility booking and proactive nudge prediction."
)

DATA_PATH = "output/prediction_review_table.csv"
METRICS_PATH = "output/metrics_summary.json"


@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.DataFrame()

    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

    return df, metrics


df, metrics = load_data()

if df.empty:
    st.warning("Prediction review table not found. Please run `python -m src.cli --mode all` first.")
    st.stop()

# -------------------------------------------------------------
# 1. Top Metrics Banner
# -------------------------------------------------------------
st.subheader("📊 Chronological Holdout Evaluation (Month 6 Unseen Test Set)")
m_cols = st.columns(6)
m_cols[0].metric("Facility Accuracy", f"{metrics.get('facility_accuracy', 0.0)*100:.1f}%")
m_cols[1].metric("Day Accuracy", f"{metrics.get('usage_day_accuracy', 0.0)*100:.1f}%")
m_cols[2].metric("Hour ±1h Acc", f"{metrics.get('usage_hour_within_1hr_accuracy', 0.0)*100:.1f}%")
m_cols[3].metric("Proactive Nudge Rate", f"{metrics.get('nudge_proactive_actionability_rate', 0.0)*100:.1f}%")
m_cols[4].metric("Exact 4/4 Matches", f"{metrics.get('exact_4_of_4_match_rate', 0.0)*100:.1f}%")
m_cols[5].metric("Avg Outputs Matched", f"{metrics.get('average_outputs_matched', 0.0):.2f} / 4.0")

st.markdown("---")

# -------------------------------------------------------------
# 2. Main Prediction Review Table (Conforming to PDF §3.5)
# -------------------------------------------------------------
st.subheader("📋 Prediction Review Output (Assignment §3.4 & §3.5)")
st.markdown(
    "Comparing model predictions against actual unseen holdout bookings with match indicators."
)

f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
search_query = f_col1.text_input("🔍 Search by Resident ID or text", "")
match_filter = f_col2.selectbox("Filter Match Status", ["All", "YES (4 of 4 Exact)", "NO (Partial Match)"])
score_filter = f_col3.selectbox("Filter by Score", ["All", "4 of 4", "3 of 4", "2 of 4", "1 of 4"])

filtered_df = df.copy()

if search_query:
    q = search_query.upper()
    filtered_df = filtered_df[
        filtered_df["Resident Reference"].str.upper().str.contains(q)
        | filtered_df["PREDICTION"].str.upper().str.contains(q)
        | filtered_df["ACTUAL"].str.upper().str.contains(q)
    ]

if match_filter == "YES (4 of 4 Exact)":
    filtered_df = filtered_df[filtered_df["MATCH"].str.contains("YES")]
elif match_filter == "NO (Partial Match)":
    filtered_df = filtered_df[filtered_df["MATCH"].str.contains("NO")]

if score_filter != "All":
    filtered_df = filtered_df[filtered_df["Score"] == score_filter]

st.dataframe(
    filtered_df[
        [
            "Resident Reference",
            "PAST BOOKINGS",
            "PREDICTION",
            "ACTUAL",
            "MATCH",
            "Score",
            "Facility Match",
            "Day Match",
            "Time Match",
            "Nudge Match",
        ]
    ],
    use_container_width=True,
    height=450,
)

st.markdown(f"*Showing {len(filtered_df)} of {len(df)} holdout records.*")

st.markdown("---")

# -------------------------------------------------------------
# 3. Resident Profile Deep Dive
# -------------------------------------------------------------
st.subheader("🔎 Resident Prediction Deep-Dive")
selected_resident = st.selectbox("Select a Resident Reference:", sorted(df["Resident Reference"].unique()))

if selected_resident:
    res_rows = df[df["Resident Reference"] == selected_resident]
    st.write(f"### Analysis for **{selected_resident}**")
    for idx, row in res_rows.iterrows():
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"**PAST BOOKINGS**\n\n```\n{row['PAST BOOKINGS']}\n```")
        with c2:
            st.success(f"**PREDICTION**\n\n```\n{row['PREDICTION']}\n```")
        with c3:
            st.warning(f"**ACTUAL RESULT**\n\n```\n{row['ACTUAL']}\n```\n**Match Score**: {row['Score']}")
