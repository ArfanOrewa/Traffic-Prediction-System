"""
pages/1_📊_Analytics.py — Historical traffic analytics (Streamlit multipage).

Auto-discovered by Streamlit as the "Analytics" page:
  - peak-hour demand curve (avg vehicle count per hour)
  - busiest road segments (top-10 by total vehicle count)
  - weekday congestion mix (stacked bar of Low/Medium/High share)
  - model comparison table (Random Forest vs XGBoost vs Logistic Regression)

Run the main app; this page appears in the sidebar navigation:
    streamlit run dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "traffic_data.csv"
COMP_PATH = BASE_DIR / "models" / "model_comparison.json"

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
CONGESTION_COLORS = {"Low": "#2ca02c", "Medium": "#e6a817", "High": "#d62728"}


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


def main() -> None:
    st.set_page_config(page_title="Traffic Analytics", page_icon="📊",
                       layout="wide")
    st.title("📊 Traffic Analytics")
    st.caption("Historical patterns from the synthetic traffic dataset")

    df = load_data()
    if df.empty:
        st.warning("No data found. Run `python generate_data.py` first.")
        return

    # --- peak hours -----------------------------------------------------------
    st.subheader("Peak-hour demand curve")
    hourly = (df.groupby("hour")["vehicle_count"].mean()
                .reset_index().sort_values("hour"))
    st.line_chart(hourly.set_index("hour")["vehicle_count"])
    peak = hourly.loc[hourly["vehicle_count"].idxmax()]
    st.caption(f"Busiest hour: {int(peak['hour']):02d}:00 "
               f"(avg {peak['vehicle_count']:.0f} vehicles)")

    # --- busiest segments -------------------------------------------------------
    st.subheader("Busiest road segments (top 10)")
    busiest = (df.groupby("road_id")["vehicle_count"].sum()
                 .sort_values(ascending=False).head(10))
    st.bar_chart(busiest)

    # --- weekday congestion mix ----------------------------------------------------
    st.subheader("Congestion mix by day of week")
    mix = (pd.crosstab(df["day_of_week"], df["congestion_level"],
                       normalize="index") * 100)
    mix = mix.reindex(DAY_ORDER)
    for level in ["Low", "Medium", "High"]:
        if level not in mix.columns:
            mix[level] = 0.0
    st.bar_chart(mix[["Low", "Medium", "High"]], color=[
        CONGESTION_COLORS["Low"], CONGESTION_COLORS["Medium"],
        CONGESTION_COLORS["High"]])

    # --- model comparison ------------------------------------------------------------
    st.subheader("Model comparison (20% hold-out set)")
    if COMP_PATH.exists():
        comp = json.load(open(COMP_PATH))
        st.dataframe(pd.DataFrame(comp), use_container_width=True,
                     hide_index=True)
        best = max(comp, key=lambda r: r["accuracy"])
        st.caption(f"Deployed model: Random Forest "
                   f"(accuracy {best['accuracy']:.4f} on the test set). "
                   f"Macro-averaged precision / recall / F1 shown.")
    else:
        st.info("Train the models first: `python model.py`")


if __name__ == "__main__":
    main()
