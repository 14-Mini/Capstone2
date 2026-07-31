"""
Smoke test for the dictionary_attack path against the live /predict endpoint.

fastapi_test_report.png exercised brute_force -> credential_stuffing (many
failed logins, seconds apart, from one IP against distinct usernames).
test_account_takeover_smoke.py exercised account_takeover. dictionary_attack
is the one attack_type in classify_attack()'s priority order
(scripts/label_and_split_v2.py) that has never been exercised against the
live, online feature computation in app.py.

classify_attack()'s dictionary_attack rule:
    fail_count_5min >= 3 AND fail_count_1min <= 2 AND unique_usernames_5min <= 8
i.e. a slow, spread-out string of failures against a small set of accounts --
enough failures to accumulate over 5 minutes, but never bursting past 2 in any
single 1-minute window (which is what would instead trip brute_force's
fail_count_1min >= 3 rule first, since brute_force is checked before
dictionary_attack).

Scenario: for each candidate gap, a fresh synthetic IP + single fixed
username sends 4 failed login attempts spaced by that gap. Same IP/user
throughout, so unique_usernames_5min is always 1 (never near the
credential_stuffing or brute_force user-count ceilings) -- the only variable
under test is how tightly spaced the failures are. We sweep the gap from
well under 60s (should stay brute_force, since fail_count_1min will hit 3+)
up past 60s (should flip to dictionary_attack, since fail_count_1min resets
below the 1-minute window on each new attempt while fail_count_5min keeps
accumulating). This confirms the deployed RF model's dictionary_attack
probability actually responds at the same boundary the offline label
threshold does.
"""
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

HOST = "127.0.0.1"
PORT = 8101
BASE_URL = f"http://{HOST}:{PORT}"

# Gap (seconds) between consecutive failed attempts from the same IP/user,
# straddling the 60s boundary between fail_count_1min and fail_count_5min.
GAP_SECONDS = [10, 20, 30, 45, 61, 75, 90, 120, 180]

BASE_TS = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)


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


def send_event(user_id, ip, ts):
    payload = {
        "ip_address": ip,
        "user_id": user_id,
        "country": "US",
        "device_type": "desktop",
        "browser_name_and_version": "Chrome 120.0",
        "os_name_and_version": "Windows 10",
        "login_successful": False,
        "login_timestamp": ts.isoformat(),
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def sweep_gap(gap_s, idx):
    # Fresh IP/user per gap so each run starts with empty rolling history.
    ip = f"203.0.113.{50 + idx}"
    user = f"dict_user_{idx}"
    ts = BASE_TS
    last_resp = None
    for _ in range(4):
        last_resp = send_event(user, ip, ts)
        ts = ts + timedelta(seconds=gap_s)
    return last_resp


def main():
    proc = start_server()
    try:
        rows = []
        for idx, gap_s in enumerate(GAP_SECONDS):
            resp = sweep_gap(gap_s, idx)
            rows.append({
                "gap_seconds": gap_s,
                "fail_count_1min": resp["features"]["fail_count_1min"],
                "fail_count_5min": resp["features"]["fail_count_5min"],
                "unique_usernames_5min": resp["features"]["unique_usernames_5min"],
                "attack_type": resp["attack_type"],
                "p_dictionary_attack": resp["attack_type_probabilities"].get("dictionary_attack", 0.0),
                "p_brute_force": resp["attack_type_probabilities"].get("brute_force", 0.0),
            })

        result = pd.DataFrame(rows)
        print("Dictionary-attack sweep (4 failed logins/IP+user, gap between attempts varying):")
        print(result.to_string(index=False))
        result.to_csv("reports/dictionary_attack_smoke_results.csv", index=False)
        print("\nSaved: reports/dictionary_attack_smoke_results.csv")

    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
