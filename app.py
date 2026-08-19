import pickle
import re
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

RF_MODEL_FILE = "models/random_forest.pkl"

WINDOW_1MIN = 60
WINDOW_5MIN = 5 * 60
WINDOW_10MIN = 10 * 60

NO_PREVIOUS_LOGIN = 999999
ALERT_FEED_MAXLEN = 500

app = FastAPI(title="RBA Attack Detection API")


def strip_version(value):
    match = re.match(r'^([A-Za-z ]+)', str(value))
    return match.group(1).strip() if match else str(value)


def device_fingerprint(device_type, browser_name_and_version, os_name_and_version):
    return f"{device_type}|{strip_version(browser_name_and_version)}|{strip_version(os_name_and_version)}"


def count_fails_successes(window):
    fail_count = sum(1 for h in window if not h[2])
    success_count = sum(1 for h in window if h[2])
    return fail_count, success_count


def login_success_rate(fail_count, success_count):
    total = fail_count + success_count
    return (success_count / total) if total > 0 else 1.0


def fail_to_success_ratio(fail_count, success_count):
    if success_count > 0:
        return min(fail_count / success_count, 999)
    return fail_count


def geo_anomaly_flag(event_country, country_counts):
    typical_country = country_counts.most_common(1)
    return 1 if (typical_country and event_country != typical_country[0][0]) else 0


class SessionStore:
    def __init__(self):
        self.ip_history = defaultdict(list)  # ip -> list of (ts, user_id, success)
        self.user_last_login = {}            # user_id -> ts
        self.user_devices = defaultdict(set)  # user_id -> set of fingerprints
        self.user_country_counts = defaultdict(Counter)  # user_id -> Counter(country)
        self.alerts = deque(maxlen=ALERT_FEED_MAXLEN)
        self._next_alert_id = 1

    def prune_ip_history(self, ip, now_ts):
        cutoff = now_ts - WINDOW_10MIN
        self.ip_history[ip] = [h for h in self.ip_history[ip] if h[0] >= cutoff]

    def compute_features(self, event, now_ts):
        user, ip = event.user_id, event.ip_address

        self.prune_ip_history(ip, now_ts)
        history = self.ip_history[ip]

        recent_1min = [h for h in history if h[0] >= now_ts - WINDOW_1MIN]
        recent_5min = [h for h in history if h[0] >= now_ts - WINDOW_5MIN]

        fail_count_1min, success_count_1min = count_fails_successes(recent_1min)
        fail_count_5min, success_count_5min = count_fails_successes(recent_5min)
        fail_count_10min, _ = count_fails_successes(history)

        users_in_window = {h[1] for h in recent_5min}
        users_in_window.add(user)

        if user in self.user_last_login:
            time_since_last_login = now_ts - self.user_last_login[user]
        else:
            time_since_last_login = NO_PREVIOUS_LOGIN

        fingerprint = device_fingerprint(
            event.device_type, event.browser_name_and_version, event.os_name_and_version
        )
        is_new_device = 0 if fingerprint in self.user_devices[user] else 1

        dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)

        return {
            'time_of_day': dt.hour,
            'ip_reputation_score': 0.0,  # not yet populated from a real IP reputation source
            'geo_anomaly_flag': geo_anomaly_flag(event.country, self.user_country_counts[user]),
            'fail_count_1min': fail_count_1min,
            'fail_count_5min': fail_count_5min,
            'unique_usernames_5min': len(users_in_window),
            'rolling_fail_velocity': fail_count_10min / 10.0,
            'login_success_rate_1min': login_success_rate(fail_count_1min, success_count_1min),
            'time_since_last_login': time_since_last_login,
            'fail_to_success_ratio_5min': fail_to_success_ratio(fail_count_5min, success_count_5min),
            'is_new_device': is_new_device,
        }, fingerprint

    def ingest(self, event, now_ts, fingerprint):
        ip, user = event.ip_address, event.user_id
        self.ip_history[ip].append((now_ts, user, event.login_successful))
        self.prune_ip_history(ip, now_ts)
        self.user_last_login[user] = now_ts
        self.user_country_counts[user][event.country] += 1
        self.user_devices[user].add(fingerprint)

    def record_alert(self, event, now_ts, attack_type, probabilities, features):
        alert = {
            "id": self._next_alert_id,
            "timestamp": datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),
            "user_id": event.user_id,
            "ip_address": event.ip_address,
            "country": event.country,
            "device_type": event.device_type,
            "login_successful": event.login_successful,
            "attack_type": attack_type,
            "attack_type_probabilities": probabilities,
            "features": features,
        }
        self._next_alert_id += 1
        self.alerts.appendleft(alert)
        return alert


store = SessionStore()


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


@app.on_event("startup")
def startup():
    global rf_data
    with open(RF_MODEL_FILE, 'rb') as f:
        rf_data = pickle.load(f)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": rf_data is not None,
        "users_tracked": len(store.user_last_login),
        "ips_tracked": len(store.ip_history),
        "alerts_in_feed": len(store.alerts),
    }


@app.get("/alerts")
def alerts(limit: int = 100):
    limit = max(1, min(limit, ALERT_FEED_MAXLEN))
    return {"count": len(store.alerts), "alerts": list(store.alerts)[:limit]}


@app.post("/predict")
def predict(event: LoginEvent):
    now_ts = event.login_timestamp.timestamp()
    features, fingerprint = store.compute_features(event, now_ts)

    feature_row = pd.DataFrame([features])[rf_data['feature_cols']]
    rf_model = rf_data['model']
    attack_type = rf_model.predict(feature_row)[0]
    probabilities = {
        cls: float(p) for cls, p in zip(rf_model.classes_, rf_model.predict_proba(feature_row)[0])
    }

    store.record_alert(event, now_ts, attack_type, probabilities, features)

    store.ingest(event, now_ts, fingerprint)

    return {"attack_type": attack_type, "attack_type_probabilities": probabilities, "features": features}
