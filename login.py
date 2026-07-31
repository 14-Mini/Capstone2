import ipaddress
import os
import sqlite3
from datetime import datetime, timezone

import requests
from flask import Flask, render_template_string, request

app = Flask(__name__)
DB_FILE = "data/users.db"
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://127.0.0.1:8000")
FALLBACK_COUNTRY = "US"  # used when the IP can't be geolocated (private/loopback, or lookup failure)

_geo_cache = {}


def geolocate_country(ip):
    """Best-effort IP -> ISO country code via ip-api.com (free, no key). Private/loopback
    addresses (e.g. 127.0.0.1 when testing locally) can never be geolocated -- that's an
    inherent limitation of local testing, not a bug in this lookup."""
    if ip in _geo_cache:
        return _geo_cache[ip]

    country = FALLBACK_COUNTRY
    try:
        addr = ipaddress.ip_address(ip)
        if not (addr.is_private or addr.is_loopback or addr.is_reserved):
            resp = requests.get(
                f"http://ip-api.com/json/{ip}", params={"fields": "countryCode"}, timeout=2
            )
            code = resp.json().get("countryCode")
            if code:
                country = code
    except (ValueError, requests.exceptions.RequestException):
        pass

    _geo_cache[ip] = country
    return country


def parse_user_agent(ua):
    device_type = "mobile" if "Mobi" in ua else "desktop"
    browser = next((b for b in ["Edge", "Chrome", "Firefox", "Safari"] if b in ua), "Other")
    os_name = next((o for o in ["Windows", "Mac OS", "Android", "iOS", "Linux"] if o in ua), "Other")
    return device_type, browser, os_name

TEMPLATE = """
<!doctype html>
<title>Login</title>
<style>
body { font-family: sans-serif; background: #fff; }
form { width: 200px; margin: 100px auto; }
input, button { display: block; width: 100%; margin-top: 8px; }
</style>
<form method="post">
<input name="username" placeholder="Username" required>
<input name="password" type="password" placeholder="Password" required>
<button type="submit">Log in</button>
{% if message %}<p>{{ message }}</p>{% endif %}
</form>
"""


@app.route("/", methods=["GET", "POST"])
def login():
    message = None
    if request.method == "POST":
        row = sqlite3.connect(DB_FILE).execute(
            "SELECT password FROM users WHERE username = ?", (request.form["username"],)
        ).fetchone()
        login_successful = bool(row and row[0] == request.form["password"])
        message = "Login successful." if login_successful else "Invalid username or password."

        device_type, browser, os_name = parse_user_agent(request.headers.get("User-Agent", ""))
        payload = {
            "ip_address": request.remote_addr,
            "user_id": request.form["username"],
            "country": geolocate_country(request.remote_addr),
            "device_type": device_type,
            "browser_name_and_version": browser,
            "os_name_and_version": os_name,
            "login_successful": login_successful,
            "login_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            requests.post(f"{APP_BASE_URL}/predict", json=payload, timeout=3)
        except requests.exceptions.RequestException:
            pass  # monitoring only — a down/unreachable model backend must never block login
    return render_template_string(TEMPLATE, message=message)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
