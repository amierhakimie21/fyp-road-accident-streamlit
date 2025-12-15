# =====================================================
# ROAD ACCIDENT PREDICTION SYSTEM (FYP-READY & VALID)
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
st.set_page_config(
    page_title="Road Accident Prediction",
    layout="wide"
)

st.title("🚗 Road Accident Prediction System")
st.write(
    "This system predicts future road accident trends using machine learning "
    "based on historical accident data."
)

# =====================================================
# LOAD DATA (STREAMLIT CLOUD SAFE)
# =====================================================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "prediction_results.csv")
    return pd.read_csv(file_path)

df = load_data()

# =====================================================
# PREPARE YEARLY DATA (FIXED)
# =====================================================
yearly = (
    df.rename(columns={
        "Predicted_Accidents": "accident_count"
    })
    .sort_values("year")
)

X = yearly[["year"]].values
y = yearly["accident_count"].values

# =====================================================
# SCALE YEAR
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# LOO CROSS VALIDATION
# =====================================================
loo = LeaveOneOut()

lr_errors, poly_errors, rf_errors = [], [], []

for train_idx, test_idx in loo.split(X_scaled):
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    lr_errors.append(mean_absolute_error(y_te, lr.predict(X_te)))

    # Polynomial Regression
    poly = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lr", LinearRegression())
    ])
    poly.fit(X_tr, y_tr)
    poly_errors.append(mean_absolute_error(y_te, poly.predict(X_te)))

    # Random Forest (comparison only)
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )
    rf.fit(X_tr, y_tr)
    rf_errors.append(mean_absolute_error(y_te, rf.predict(X_te)))

mae_lr = np.mean(lr_errors)
mae_poly = np.mean(poly_errors)
mae_rf = np.mean(rf_errors)

# =====================================================
# MODEL PERFORMANCE
# =====================================================
st.subheader("📌 Model Performance (MAE – Leave-One-Out CV)")

c1, c2, c3 = st.columns(3)
c1.metric("Linear Regression", f"{mae_lr:,.2f}")
c2.metric("Polynomial Regression", f"{mae_poly:,.2f}")
c3.metric("Random Forest", f"{mae_rf:,.2f}")

# =====================================================
# MODEL INTERPRETATION
# =====================================================
st.subheader("🧠 Model Interpretation")

st.write("""
- **Linear Regression** serves as a baseline model for interpretability.
- **Polynomial Regression (degree = 2)** captures non-linear temporal trends and is used for future prediction.
- **Random Forest** is evaluated for comparison but is not suitable for extrapolating unseen future years.
- **Leave-One-Out Cross Validation (LOO-CV)** is applied due to the limited number of yearly observations.
""")

# =====================================================
# TRAIN FINAL MODELS
# =====================================================
lr_final = LinearRegression()
lr_final.fit(X_scaled, y)

poly_final = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("lr", LinearRegression())
])
poly_final.fit(X_scaled, y)

yearly["LR_Pred"] = lr_final.predict(X_scaled)
yearly["Poly_Pred"] = poly_final.predict(X_scaled)

# =====================================================
# FUTURE PREDICTION
# =====================================================
st.subheader("📊 Future Accident Prediction")

future_year = st.slider(
    "Select year to predict",
    min_value=int(yearly["year"].max()) + 1,
    max_value=int(yearly["year"].max()) + 10,
    value=int(yearly["year"].max()) + 1
)

future_scaled = scaler.transform([[future_year]])
future_prediction = int(poly_final.predict(future_scaled)[0])

st.success(
    f"Predicted accidents in {future_year} "
    f"(Polynomial Regression): **{future_prediction:,} cases**"
)

# =====================================================
# MULTI-YEAR FORECAST
# =====================================================
st.subheader("📅 Multi-Year Accident Forecast")

future_years = pd.DataFrame({
    "Year": range(
        int(yearly["year"].max()) + 1,
        int(yearly["year"].max()) + 6
    )
})

future_scaled_all = scaler.transform(future_years[["Year"]].values)
future_years["Predicted Accidents"] = (
    poly_final.predict(future_scaled_all).astype(int)
)

st.dataframe(future_years, use_container_width=True)

# =====================================================
# VISUALIZATION
# =====================================================
st.subheader("📈 Accident Trend & Prediction")

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(yearly["year"], yearly["accident_count"], marker="o", label="Actual")
ax.plot(yearly["year"], yearly["LR_Pred"], linestyle="--", label="Linear Regression")
ax.plot(yearly["year"], yearly["Poly_Pred"], linestyle="-.", label="Polynomial Regression")

ax.plot(
    future_years["Year"],
    future_years["Predicted Accidents"],
    marker="o",
    linestyle="--",
    color="red",
    label="Future Prediction"
)

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
- Predictions are based solely on historical accident trends.
- Limited yearly observations may restrict long-term accuracy.
- External factors such as weather, road conditions, or policy changes are not included.
- Results are intended strictly for academic and analytical purposes.
""")
