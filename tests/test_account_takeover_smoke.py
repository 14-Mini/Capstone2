"""
Smoke test for the account_takeover path against the live /predict endpoint.

fastapi_test_report.png only exercised brute_force (many failed logins, one IP).
account_takeover is the RF model's weakest class offline (rf_report.txt: 0.079
precision, 0.381 recall) and is checked FIRST in classify_attack()'s priority
order, so it's the class most worth confirming the deployed model actually
recognizes before it's written up.

Scenario: one real seeded user (from full_labeled.csv) logs in successfully from
a new country and a new device fingerprint, at increasing gaps since their last
known login. classify_attack()'s training-time rule fires at
time_since_last_login > 24h (ATO_TIME_GAP=86400s) AND geo_anomaly_flag=1 AND
is_new_device=1 AND login_successful. This test sweeps the gap across that 24h
boundary and checks whether the *deployed* RF model's account_takeover
probability actually responds where the hand-tuned label threshold says it
should.

Requests are sent sequentially with increasing login_timestamp, so each
request's own gap-since-last-login is exactly (this timestamp - previous
request's timestamp) -- the live store's own state chains the gaps correctly
without needing to reset anything between calls.
"""
import pickle
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

HOST = "127.0.0.1"
PORT = 8100
BASE_URL = f"http://{HOST}:{PORT}"

# Sweep gaps (hours since last login), straddling ATO_TIME_GAP = 24h.
GAP_HOURS = [1, 6, 12, 18, 22, 24, 26, 30, 48, 120]


def pick_seed_user():
    df = pd.read_csv("data/splits/full_labeled.csv", low_memory=False)
    df["login_timestamp"] = pd.to_datetime(df["login_timestamp"], errors="coerce")
    counts = df.groupby("user_id").size()
    for uid in counts[counts >= 5].index:
        sub = df[df["user_id"] == uid].sort_values("login_timestamp")
        if sub["country"].nunique() == 1 and sub["login_successful"].sum() >= 1:
            last = sub.iloc[-1]
            return {
                "user_id": str(uid),
                "last_login": last["login_timestamp"].to_pydatetime().replace(tzinfo=timezone.utc),
                "home_country": last["country"],
                "home_device_type": last["device_type"],
                "home_browser": last["browser_name_and_version"],
                "home_os": last["os_name_and_version"],
            }
    raise RuntimeError("no suitable seed user found")


def start_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", HOST, "--port", str(PORT)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for _ in range(60):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=1)
            if r.ok and r.json().get("models_loaded"):
                return proc
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("server did not become healthy in time")


def send_event(user_id, ip, country, device_type, browser, os_name, success, ts):
    payload = {
        "ip_address": ip,
        "user_id": user_id,
        "country": country,
        "device_type": device_type,
        "browser_name_and_version": browser,
        "os_name_and_version": os_name,
        "login_successful": success,
        "login_timestamp": ts.isoformat(),
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def main():
    seed = pick_seed_user()
    print(f"Seed user {seed['user_id']}: home_country={seed['home_country']!r}, "
          f"home_device=({seed['home_device_type']}, {seed['home_browser']}, {seed['home_os']}), "
          f"last_login={seed['last_login'].isoformat()}")

    proc = start_server()
    try:
        # Control: same country, same device, small gap, success -> should NOT look like ATO.
        control = send_event(
            seed["user_id"], "203.0.113.10", seed["home_country"],
            seed["home_device_type"], seed["home_browser"], seed["home_os"],
            True, seed["last_login"] + timedelta(minutes=30),
        )
        print("\nControl (same country/device, 30min gap):")
        print(f"  attack_type={control['attack_type']!r}  "
              f"P(ato)={control['attack_type_probabilities'].get('account_takeover', 0):.3f}  "
              f"is_anomaly={control['is_anomaly']}  score={control['anomaly_score']:.3f}")

        # ATO sweep: new country, new device, success, gap climbing across 24h.
        last_ts = seed["last_login"] + timedelta(minutes=30)
        rows = []
        for gap_h in GAP_HOURS:
            ts = last_ts + timedelta(hours=gap_h)
            resp = send_event(
                seed["user_id"], "198.51.100.77", "RU",
                "desktop", "Firefox 90.0", "Windows 10",
                True, ts,
            )
            last_ts = ts
            rows.append({
                "gap_hours_requested": gap_h,
                "time_since_last_login_s": resp["features"]["time_since_last_login"],
                "geo_anomaly_flag": resp["features"]["geo_anomaly_flag"],
                "is_new_device": resp["features"]["is_new_device"],
                "attack_type": resp["attack_type"],
                "p_account_takeover": resp["attack_type_probabilities"].get("account_takeover", 0.0),
                "is_anomaly": resp["is_anomaly"],
                "anomaly_score": resp["anomaly_score"],
            })

        result = pd.DataFrame(rows)
        print("\nATO sweep (new country + new device + success, gap climbing):")
        print(result.to_string(index=False))
        result.to_csv("reports/ato_smoke_results.csv", index=False)
        print("\nSaved: reports/ato_smoke_results.csv")

        with open("models/isolation_forest.pkl", "rb") as f:
            if_data = pickle.load(f)
        offset = if_data["model"].offset_
        print(f"\nIsolation Forest decision boundary (anomaly_score domain): {-offset:.4f}")

    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
