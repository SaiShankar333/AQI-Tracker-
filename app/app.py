import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AQI Predictor", layout="wide")

# Load the updated realistic dataset
df = pd.read_csv("data/aqi_dataset.csv")
cities = df["City"].unique()

# Sidebar controls
st.sidebar.title("Settings")
selected_city = st.sidebar.selectbox("Select City", sorted(cities))
days_to_predict = st.sidebar.slider("Days to Predict", 1, 14, 7)

st.sidebar.info("""
This app shows AQI predictions and trends for Indian and global cities.

AQI Categories:
- 0–50: Good (Green)
- 51–100: Moderate (Gold)
- 101–150: Unhealthy for Sensitive Groups (Orange)
- 151–200: Unhealthy (Red)
- 201–300: Very Unhealthy (Purple)
- 301+: Hazardous (Maroon)
""")

# Filter data for the selected city
city_df = df[df["City"] == selected_city]

# AQI Category Color Logic (Gold fix)
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#00e400"
    elif aqi <= 100:
        return "Moderate", "#FFD700"  # Fixed: Gold for contrast
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"

# Train model on the full dataset
model = RandomForestRegressor(n_estimators=100, random_state=42)
X = df[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity", "WindSpeed"]]
y = df["Predicted_AQI"]
model.fit(X, y)

# Fixed: simulate_prediction with column names to avoid warning
def simulate_prediction(base, days):
    future_dates = [datetime.today() + timedelta(days=i) for i in range(1, days + 1)]
    predictions = []
    for i in range(days):
        row = base.sample(1).iloc[0]
        features = row[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity", "WindSpeed"]].to_frame().T
        aqi_pred = int(model.predict(features)[0])
        predictions.append((future_dates[i], aqi_pred))
    return predictions

# Main UI layout
st.title("🌍 AQI Predictor Dashboard")
tab1, tab2, tab3, tab4 = st.tabs(["📈 Predictions", "📊 Historical", "📋 Parameters", "⚙️ Comparison"])

# === TAB 1: Predictions ===
with tab1:
    latest_value = int(city_df["Predicted_AQI"].iloc[-1])
    category, color = get_aqi_category(latest_value)

    st.subheader(f"{selected_city}")
    st.markdown(f"**Today**: {datetime.today().strftime('%B %d, %Y')}")
    st.markdown(f"<h1 style='color:{color};'>{latest_value}</h1>", unsafe_allow_html=True)
    st.markdown(f"### {category}")

    st.markdown("### 14-Day AQI Forecast")
    pred = simulate_prediction(city_df, days_to_predict)
    for date, aqi in pred:
        cat, cat_color = get_aqi_category(aqi)
        st.markdown(
            f"<div style='background-color:{cat_color};padding:10px;margin:5px;border-radius:10px;'>"
            f"<b>{date.strftime('%a, %b %d')}</b>: {aqi} – {cat}"
            f"</div>", unsafe_allow_html=True
        )

    st.markdown("### Historical + Predicted AQI Trend")
    historical = city_df.tail(100).copy()
    historical["Date"] = pd.date_range(end=datetime.today(), periods=len(historical))

    hist_fig = px.line(historical, x="Date", y="Predicted_AQI", title="Historical AQI")
    hist_fig.update_traces(line=dict(color="blue"))

    future_df = pd.DataFrame(pred, columns=["Date", "Predicted_AQI"])
    pred_fig = px.scatter(future_df, x="Date", y="Predicted_AQI", color_discrete_sequence=["orange"], title="Predicted AQI")
    for trace in pred_fig.data:
        hist_fig.add_trace(trace)

    st.plotly_chart(hist_fig, use_container_width=True)

# === TAB 2: Historical Analysis ===
with tab2:
    st.subheader("📊 AQI Statistics & Distribution")

    df["Date"] = pd.date_range(start="2023-07-01", periods=len(df))
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    monthly_avg = df[df["City"] == selected_city].groupby("Month")["Predicted_AQI"].mean().reset_index()

    st.plotly_chart(
        px.bar(monthly_avg, x="Month", y="Predicted_AQI", color="Predicted_AQI", color_continuous_scale="Turbo"),
        use_container_width=True
    )

    st.plotly_chart(
        px.histogram(city_df, x="Predicted_AQI", nbins=40, title="AQI Distribution"),
        use_container_width=True
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min AQI", f"{int(city_df['Predicted_AQI'].min())}")
    col2.metric("Max AQI", f"{int(city_df['Predicted_AQI'].max())}")
    col3.metric("Average AQI", f"{city_df['Predicted_AQI'].mean():.1f}")
    col4.metric("Std Dev", f"{city_df['Predicted_AQI'].std():.1f}")

# === TAB 3: Parameter Trends ===
with tab3:
    st.subheader(f"{selected_city} – Parameter Overview")
    param_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity", "WindSpeed"]
    st.dataframe(city_df[param_cols].describe().T.style.format(precision=2), use_container_width=True)

    for col in param_cols:
        fig = px.line(city_df.reset_index(), y=col, title=f"{col} Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# === TAB 4: Model Comparison Placeholder ===
with tab4:
    st.info("Model comparison (Random Forest vs others) will be implemented here.")

# === FOOTER ===
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<center><small>© 2025 AQI Dashboard | Realistic data generated for educational use</small></center>", unsafe_allow_html=True)
