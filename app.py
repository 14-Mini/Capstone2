import pickle
import re
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ============================================================================
# CONFIGURATION
# ============================================================================

RF_MODEL_FILE = "random_forest.pkl"
IF_MODEL_FILE = "isolation_forest.pkl"
SEED_FILE = "full_labeled.csv"

WINDOW_1MIN = 60
WINDOW_5MIN = 5 * 60
WINDOW_10MIN = 10 * 60

NO_PREVIOUS_LOGIN = 999999

ALERT_FEED_MAXLEN = 500  # bounded live alert feed for the dashboard

app = FastAPI(title="RBA Attack Detection API")


# ============================================================================
# FEATURE HELPERS (same formulas as feature_engineering.py, computed online)
# ============================================================================

def strip_version(value):
    match = re.match(r'^([A-Za-z ]+)', str(value))
    return match.group(1).strip() if match else str(value)


def device_fingerprint(device_type, browser_name_and_version, os_name_and_version):
    return f"{device_type}|{strip_version(browser_name_and_version)}|{strip_version(os_name_and_version)}"


def most_common_country(counter):
    if not counter:
        return None
    return counter.most_common(1)[0][0]


# ============================================================================
# IN-MEMORY SESSION STORE
# ============================================================================

class SessionStore:
    def __init__(self):
        # Scope 1: short-term per-IP rolling window (pruned to 10 minutes)
        self.ip_history = defaultdict(list)  # ip -> list of (ts, user_id, success)

        # Scope 2: long-term per-user profile (persists indefinitely)
        self.user_last_login = {}            # user_id -> ts
        self.user_devices = defaultdict(set)  # user_id -> set of fingerprints
        self.user_country_counts = defaultdict(Counter)  # user_id -> Counter(country)

        # Scope 3: static global lookup, computed once at startup
        self.ip_reputation = {}              # ip -> historical mean(is_attack_ip)

        # Scope 4: bounded live alert feed, newest first (dashboard reads this)
        self.alerts = deque(maxlen=ALERT_FEED_MAXLEN)
        self._next_alert_id = 1

    def seed_from_csv(self, path):
        df = pd.read_csv(path, encoding='utf-8', low_memory=False)
        df['user_id'] = df['user_id'].astype(str)
        df['ip_address'] = df['ip_address'].astype(str)
        df['login_timestamp'] = pd.to_datetime(df['login_timestamp'], errors='coerce')
        df = df.sort_values('login_timestamp')

        self.ip_reputation = df.groupby('ip_address')['is_attack_ip'].mean().to_dict()

        for _, row in df.iterrows():
            user = row['user_id']
            ts = row['login_timestamp'].timestamp()

            self.user_last_login[user] = ts
            self.user_country_counts[user][row['country']] += 1
            fingerprint = device_fingerprint(
                row['device_type'], row['browser_name_and_version'], row['os_name_and_version']
            )
            self.user_devices[user].add(fingerprint)

    def prune_ip_history(self, ip, now_ts):
        cutoff = now_ts - WINDOW_10MIN
        self.ip_history[ip] = [h for h in self.ip_history[ip] if h[0] >= cutoff]

    def compute_features(self, event, now_ts):
        user = event.user_id
        ip = event.ip_address

        self.prune_ip_history(ip, now_ts)
        history = self.ip_history[ip]

        recent_1min = [h for h in history if h[0] >= now_ts - WINDOW_1MIN]
        recent_5min = [h for h in history if h[0] >= now_ts - WINDOW_5MIN]

        fail_count_1min = sum(1 for h in recent_1min if not h[2])
        fail_count_5min = sum(1 for h in recent_5min if not h[2])
        success_count_1min = sum(1 for h in recent_1min if h[2])
        success_count_5min = sum(1 for h in recent_5min if h[2])
        fail_count_10min = sum(1 for h in history if not h[2])

        users_in_window = set(h[1] for h in recent_5min)
        users_in_window.add(user)
        unique_usernames_5min = len(users_in_window)

        if user in self.user_last_login:
            time_since_last_login = now_ts - self.user_last_login[user]
        else:
            time_since_last_login = NO_PREVIOUS_LOGIN

        typical_country = most_common_country(self.user_country_counts[user])
        geo_anomaly_flag = 1 if (typical_country is not None and event.country != typical_country) else 0

        fingerprint = device_fingerprint(
            event.device_type, event.browser_name_and_version, event.os_name_and_version
        )
        is_new_device = 0 if fingerprint in self.user_devices[user] else 1

        rolling_fail_velocity = fail_count_10min / 10.0

        total_1min = fail_count_1min + success_count_1min
        login_success_rate_1min = (success_count_1min / total_1min) if total_1min > 0 else 1.0

        if success_count_5min > 0:
            fail_to_success_ratio_5min = fail_count_5min / success_count_5min
        else:
            fail_to_success_ratio_5min = fail_count_5min
        fail_to_success_ratio_5min = min(fail_to_success_ratio_5min, 999)

        dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)

        return {
            'time_of_day': dt.hour,
            'ip_reputation_score': self.ip_reputation.get(ip, 0.0),
            'geo_anomaly_flag': geo_anomaly_flag,
            'fail_count_1min': fail_count_1min,
            'fail_count_5min': fail_count_5min,
            'unique_usernames_5min': unique_usernames_5min,
            'rolling_fail_velocity': rolling_fail_velocity,
            'login_success_rate_1min': login_success_rate_1min,
            'time_since_last_login': time_since_last_login,
            'fail_to_success_ratio_5min': fail_to_success_ratio_5min,
            'is_new_device': is_new_device,
        }, fingerprint

    def ingest(self, event, now_ts, fingerprint):
        ip = event.ip_address
        user = event.user_id

        self.ip_history[ip].append((now_ts, user, event.login_successful))
        self.prune_ip_history(ip, now_ts)

        self.user_last_login[user] = now_ts
        self.user_country_counts[user][event.country] += 1
        self.user_devices[user].add(fingerprint)

    def record_alert(self, event, now_ts, result):
        alert = {
            "id": self._next_alert_id,
            "timestamp": datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),
            "user_id": event.user_id,
            "ip_address": event.ip_address,
            "country": event.country,
            "device_type": event.device_type,
            "login_successful": event.login_successful,
            "attack_type": result["attack_type"],
            "attack_type_probabilities": result["attack_type_probabilities"],
            "is_anomaly": result["is_anomaly"],
            "anomaly_score": result["anomaly_score"],
            "features": result["features"],
        }
        self._next_alert_id += 1
        self.alerts.appendleft(alert)  # deque prunes oldest from the right at maxlen
        return alert


store = SessionStore()


# ============================================================================
# MODELS
# ============================================================================

class LoginEvent(BaseModel):
    ip_address: str
    user_id: str
    country: str
    device_type: str
    browser_name_and_version: str
    os_name_and_version: str
    login_successful: bool
    login_timestamp: datetime


rf_data = None
if_data = None


@app.on_event("startup")
def startup():
    global rf_data, if_data

    with open(RF_MODEL_FILE, 'rb') as f:
        rf_data = pickle.load(f)
    with open(IF_MODEL_FILE, 'rb') as f:
        if_data = pickle.load(f)

    store.seed_from_csv(SEED_FILE)
    print(f"Seeded store: {len(store.user_last_login):,} users, "
          f"{len(store.ip_reputation):,} IPs")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": rf_data is not None and if_data is not None,
        "users_tracked": len(store.user_last_login),
        "ips_tracked": len(store.ip_history),
        "alerts_in_feed": len(store.alerts),
    }


@app.get("/alerts")
def alerts(limit: int = 100):
    limit = max(1, min(limit, ALERT_FEED_MAXLEN))
    return {"count": len(store.alerts), "alerts": list(store.alerts)[:limit]}


@app.get("/session/{user_id}")
def session_debug(user_id: str, ip_address: str | None = None):
    result = {
        "user_id": user_id,
        "last_login": store.user_last_login.get(user_id),
        "devices_seen": list(store.user_devices.get(user_id, set())),
        "country_counts": dict(store.user_country_counts.get(user_id, {})),
    }
    if ip_address:
        result["ip_history"] = store.ip_history.get(ip_address, [])
        result["ip_reputation"] = store.ip_reputation.get(ip_address, 0.0)
    return result


@app.post("/predict")
def predict(event: LoginEvent):
    now_ts = event.login_timestamp.timestamp()

    features, fingerprint = store.compute_features(event, now_ts)

    feature_row = pd.DataFrame([features])

    rf_model = rf_data['model']
    rf_row = feature_row[rf_data['feature_cols']]
    attack_type = rf_model.predict(rf_row)[0]
    proba = rf_model.predict_proba(rf_row)[0]
    attack_type_probabilities = {
        cls: float(p) for cls, p in zip(rf_model.classes_, proba)
    }

    if_model = if_data['model']
    if_row = feature_row[if_data['feature_cols']]
    is_anomaly = bool(if_model.predict(if_row)[0] == -1)
    anomaly_score = float(-if_model.score_samples(if_row)[0])

    result = {
        "attack_type": attack_type,
        "attack_type_probabilities": attack_type_probabilities,
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "features": features,
    }

    if is_anomaly or attack_type != "normal":
        store.record_alert(event, now_ts, result)

    store.ingest(event, now_ts, fingerprint)

    return result
