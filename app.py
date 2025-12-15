# =====================================================
# ROAD ACCIDENT TREND ANALYSIS SYSTEM (FYP-READY & HOSTING-SAFE)
# =====================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Road Accident Trend Analysis",
    layout="wide"
)

st.title("🚗 Road Accident Trend Analysis System")
st.write(
    "This system illustrates road accident trend behaviour using machine learning "
    "based on yearly accident data."
)
st.caption("All analyses are based on data starting from year 2023.")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "prediction_results.csv")

    if not os.path.exists(file_path):
        st.error("❌ prediction_results.csv not found.")
        st.stop()

    df = pd.read_csv(file_path)

    if not {"year", "Predicted_Accidents"}.issubset(df.columns):
        st.error("❌ CSV must contain columns: year, Predicted_Accidents")
        st.stop()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["Predicted_Accidents"] = pd.to_numeric(df["Predicted_Accidents"], errors="coerce")

    return df.dropna()

df = load_data()

# =====================================================
# PREPARE DATA
# =====================================================
yearly = (
    df.rename(columns={"Predicted_Accidents": "accident_count"})
    .query("year >= 2023")
    .sort_values("year")
)

if len(yearly) < 3:
    st.error("❌ Not enough yearly data for modelling.")
    st.stop()

X = yearly[["year"]].values
y = yearly["accident_count"].values

# =====================================================
# SCALE + TRAIN MODEL
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("lr", LinearRegression())
])
poly_model.fit(X_scaled, y)

# =====================================================
# YEAR SLICER
# =====================================================
st.subheader("🎚 Select Year for Trend Illustration")

selected_year = st.slider(
    "Choose a year",
    min_value=2023,
    max_value=2032,
    value=2028,
    step=1
)

# =====================================================
# BUTTON
# =====================================================
show_result = st.button("▶ Show Result")

# =====================================================
# RESULT SECTION
# =====================================================
if show_result:
    st.markdown("---")
    st.subheader("📊 Model-Generated Result")

    # Prediction
    scaled_year = scaler.transform([[selected_year]])
    predicted_value = int(poly_model.predict(scaled_year)[0])

    result_df = pd.DataFrame({
        "Year": [selected_year],
        "Predicted Accidents": [predicted_value]
    })

    st.dataframe(result_df, use_container_width=True)

    st.success(
        f"Model-generated accident count for **{selected_year}** is "
        f"**{predicted_value:,} cases**."
    )

    # =================================================
    # VISUALIZATION (BEST PRACTICE)
    # =================================================
    st.subheader("📈 Trend Context Visualization")

    fig, ax = plt.subplots(figsize=(10, 4))

    # Historical data
    ax.plot(
        yearly["year"],
        yearly["accident_count"],
        marker="o",
        label="Historical Data (≥2023)"
    )

    # Extrapolation line (dashed)
    last_year = yearly["year"].iloc[-1]
    last_value = yearly["accident_count"].iloc[-1]

    ax.plot(
        [last_year, selected_year],
        [last_value, predicted_value],
        linestyle=":",
        color="red",
        alpha=0.7,
        label="Extrapolated Trend"
    )

    # Selected prediction point
    ax.scatter(
        selected_year,
        predicted_value,
        color="red",
        s=100,
        zorder=5,
        label="Selected Year (Model Output)"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Accident Count")
    ax.set_xlim(2023, max(2032, selected_year))
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.info(
        "The dashed red line represents model extrapolation beyond observed data. "
        "Values are generated for trend illustration purposes only."
    )

# =====================================================
# LIMITATIONS
# =====================================================
st.subheader("⚠ Model Limitations")
st.info("""
- The dataset contains a limited number of yearly observations.
- Generated values are illustrative and do not represent actual future accident counts.
- External factors such as policy changes, weather, and infrastructure are not included.
- Results are intended for academic and analytical purposes only.
""")
