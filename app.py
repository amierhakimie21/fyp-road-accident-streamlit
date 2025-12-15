# =====================================================
# ROAD ACCIDENT PREDICTION SYSTEM (FYP-READY & HOSTING-SAFE)
# =====================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Road Accident Prediction", layout="wide")

st.title("🚗 Road Accident Prediction System")
st.write(
    "This system illustrates road accident trend behaviour using machine learning "
    "based on yearly accident data."
)
st.caption(
    "All analyses and model-generated trends are conducted using data starting from year 2023."
)

# =====================================================
# LOAD DATA (FAIL-SAFE)
# =====================================================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "prediction_results.csv")

    if not os.path.exists(file_path):
        st.error("❌ prediction_results.csv not found in repository.")
        st.stop()

    df = pd.read_csv(file_path)

    required_cols = {"year", "Predicted_Accidents"}
    if not required_cols.issubset(df.columns):
        st.error("❌ CSV must contain columns: year, Predicted_Accidents")
        st.stop()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["Predicted_Accidents"] = pd.to_numeric(df["Predicted_Accidents"], errors="coerce")
    df = df.dropna()

    return df

df = load_data()

# =====================================================
# PREPARE DATA (2023 ONWARDS)
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
# SCALE YEAR
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# MODEL BEHAVIOUR ANALYSIS (LOO-CV)
# =====================================================
loo = LeaveOneOut()
lr_errors, poly_errors, rf_errors = [], [], []

for train_idx, test_idx in loo.split(X_scaled):
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    lr = LinearRegression().fit(X_tr, y_tr)
    lr_errors.append(mean_absolute_error(y_te, lr.predict(X_te)))

    poly = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lr", LinearRegression())
    ]).fit(X_tr, y_tr)
    poly_errors.append(mean_absolute_error(y_te, poly.predict(X_te)))

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X_tr, y_tr)
    rf_errors.append(mean_absolute_error(y_te, rf.predict(X_te)))

mae_lr = np.mean(lr_errors)
mae_poly = np.mean(poly_errors)
mae_rf = np.mean(rf_errors)

# =====================================================
# MODEL BEHAVIOUR DISPLAY
# =====================================================
st.subheader("📌 Model Behaviour Analysis (On Generated Data)")
st.caption(
    "This evaluation illustrates model behaviour on limited yearly trend data. "
    "Zero error values indicate perfect curve fitting rather than real-world predictive accuracy."
)

def show_metric(v):
    return "Perfect Fit" if v < 1e-6 else f"{v:,.2f}"

c1, c2, c3 = st.columns(3)
c1.metric("Linear Regression (Curve Fit)", show_metric(mae_lr))
c2.metric("Polynomial Regression (Curve Fit)", show_metric(mae_poly))
c3.metric("Random Forest (Comparison)", f"{mae_rf:,.2f}")

# =====================================================
# MODEL INTERPRETATION
# =====================================================
st.subheader("🧠 Model Interpretation")
st.write("""
- **Linear Regression** provides a simple and interpretable baseline.
- **Polynomial Regression (degree = 2)** captures non-linear temporal patterns and is used for trend generation.
- **Random Forest** is included for comparison but is unsuitable for future extrapolation.
- **Leave-One-Out Cross Validation (LOO-CV)** is used to demonstrate model behaviour on limited data.
""")

# =====================================================
# TRAIN FINAL MODELS
# =====================================================
lr_final = LinearRegression().fit(X_scaled, y)

poly_final = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("lr", LinearRegression())
]).fit(X_scaled, y)

yearly["Poly_Fit"] = poly_final.predict(X_scaled)

# =====================================================
# MODEL-GENERATED TREND (2023–2032)
# =====================================================
st.subheader("📊 Model-Generated Accident Trend (2023–2032)")

prediction_years = pd.DataFrame({
    "Year": range(2023, 2033)
})

prediction_scaled = scaler.transform(prediction_years[["Year"]])
prediction_years["Predicted Accidents"] = (
    poly_final.predict(prediction_scaled).astype(int)
)

st.dataframe(prediction_years, use_container_width=True)

st.info(
    "Accident values from 2023 to 2032 are generated using polynomial regression "
    "to illustrate long-term trend behaviour and do not represent actual future accident counts."
)

# =====================================================
# VISUALIZATION
# =====================================================
st.subheader("📈 Accident Trend & Model-Generated Projection")

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    yearly["year"],
    yearly["accident_count"],
    marker="o",
    label="Input Data (2023–2027)"
)

ax.plot(
    yearly["year"],
    yearly["Poly_Fit"],
    linestyle="-.",
    label="Polynomial Fit"
)

ax.plot(
    prediction_years["Year"],
    prediction_years["Predicted Accidents"],
    marker="o",
    linestyle="--",
    color="red",
    label="Model-Generated Trend (2023–2032)"
)

ax.set_xlim(2023, 2032)
ax.set_xlabel("Year")
ax.set_ylabel("Accident Count")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# =====================================================
# LIMITATIONS
# =====================================================
st.subheader("⚠ Model Limitations")
st.info("""
- The dataset contains a limited number of yearly observations.
- Generated values are intended to illustrate trend behaviour only.
- External factors such as policy changes, weather conditions, and infrastructure development are not included.
- Results are for academic and analytical purposes only.
""")
