"""
Streamlit dashboard for the RBA Attack Detection API (app.py).

12-week plan, Week 11 deliverable: live alert feed, attack map, anomaly gauge,
fail-velocity trend chart, all read from the FastAPI backend's /alerts and
/health endpoints (no direct DB/model access, no separate state).
"""
import os
import pickle
import time
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycountry
import requests
import streamlit as st


def _resolve_api_base_url():
    # Streamlit Cloud secrets take priority; falls back to an env var (e.g.
    # Render) and then localhost for local development.
    try:
        return st.secrets["API_BASE_URL"]
    except Exception:
        return os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


API_BASE_URL = _resolve_api_base_url()
IF_MODEL_FILE = "isolation_forest.pkl"

STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"
CATEGORICAL = {
    "normal": "#2a78d6",
    "brute_force": "#e34948",
    "credential_stuffing": "#eb6834",
    "dictionary_attack": "#eda100",
    "account_takeover": "#4a3aa7",
}

st.set_page_config(page_title="RBA Attack Detection Dashboard", layout="wide")


@st.cache_resource
def anomaly_decision_boundary():
    with open(IF_MODEL_FILE, "rb") as f:
        if_data = pickle.load(f)
    return float(-if_data["model"].offset_)


@st.cache_data(ttl=3600)
def alpha2_to_alpha3(code):
    try:
        return pycountry.countries.get(alpha_2=code).alpha_3
    except AttributeError:
        return None


def fetch_json(path, **params):
    r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=5)
    r.raise_for_status()
    return r.json()


st.title("RBA Attack Detection — Live Dashboard")

with st.sidebar:
    st.header("Controls")
    limit = st.slider("Alerts to fetch", min_value=20, max_value=500, value=200, step=20)
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_seconds = st.number_input("Refresh interval (s)", min_value=2, max_value=60, value=5)

try:
    health = fetch_json("/health")
    alerts_payload = fetch_json("/alerts", limit=limit)
except requests.exceptions.RequestException as exc:
    st.error(f"Cannot reach the API at {API_BASE_URL} — is `uvicorn app:app` running? ({exc})")
    st.stop()

alerts = alerts_payload["alerts"]
df = pd.DataFrame(alerts)
st.caption(f"Last updated {datetime.now(tz=timezone.utc).strftime('%H:%M:%S UTC')}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Alerts in feed", health["alerts_in_feed"])
col2.metric("Users tracked", f"{health['users_tracked']:,}")
col3.metric("IPs tracked", f"{health['ips_tracked']:,}")
col4.metric("Models loaded", "yes" if health["models_loaded"] else "no")

if df.empty:
    st.info("No alerts yet. Send some events to POST /predict on the API to populate the dashboard.")
    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["rolling_fail_velocity"] = df["features"].apply(lambda f: f["rolling_fail_velocity"])

gauge_col, dist_col = st.columns(2)

with gauge_col:
    boundary = anomaly_decision_boundary()
    latest_score = df.iloc[0]["anomaly_score"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_score,
        title={"text": "Latest anomaly score"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": STATUS_CRITICAL if latest_score >= boundary else STATUS_GOOD},
            "steps": [
                {"range": [0, boundary], "color": "#e1e0d9"},
                {"range": [boundary, 1], "color": "#f6d9cf"},
            ],
            "threshold": {
                "line": {"color": STATUS_WARNING, "width": 3},
                "value": boundary,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

with dist_col:
    counts = df["attack_type"].value_counts().reindex(CATEGORICAL.keys()).dropna()
    fig = px.bar(
        x=counts.index, y=counts.values,
        color=counts.index, color_discrete_map=CATEGORICAL,
        labels={"x": "attack_type", "y": "alerts"},
        title="Attack type distribution (current feed)",
    )
    fig.update_layout(height=300, showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Geographic attack map")
country_counts = df["country"].value_counts().rename_axis("country").reset_index(name="alerts")
country_counts["iso3"] = country_counts["country"].apply(alpha2_to_alpha3)
mapped = country_counts.dropna(subset=["iso3"])
fig = px.choropleth(
    mapped, locations="iso3", color="alerts",
    color_continuous_scale=["#cde2fb", "#3987e5", "#0d366b"],
    hover_name="country",
)
fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Fail velocity trend")
trend = df.sort_values("timestamp")
fig = px.line(trend, x="timestamp", y="rolling_fail_velocity", markers=True)
fig.update_traces(line_color="#2a78d6")
fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=20))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Live alert feed")
display_cols = ["timestamp", "user_id", "ip_address", "country", "attack_type", "is_anomaly", "anomaly_score"]
st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
