import os

import requests
from flask import Flask, render_template_string

app = Flask(__name__)
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RBA Attack Detection</title>
  <style>
    body {
      font-family: sans-serif;
      margin: 2rem;
    }
    .kpi {
      display: flex;
      gap: 2rem;
      margin-bottom: 1rem;
    }
    .kpi div {
      font-size: 1.5rem;
      font-weight: bold;
    }
    table {
      border-collapse: collapse;
      width: 100%;
    }
    th, td {
      border: 1px solid #ccc;
      padding: 4px 8px;
      text-align: left;
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <h1>RBA Attack Detection &mdash; Live Dashboard</h1>

  {% if error %}
    <p style="color:red">Cannot reach the API at {{ api_url }}: {{ error }}</p>
  {% else %}
    <div class="kpi">
      <div>Alerts<br>{{ health.alerts_in_feed }}</div>
      <div>Users<br>{{ health.users_tracked }}</div>
      <div>IPs<br>{{ health.ips_tracked }}</div>
      <div>Models loaded<br>{{ "yes" if health.models_loaded else "no" }}</div>
    </div>
    <table>
      <tr>
        <th>Time</th>
        <th>User</th>
        <th>IP</th>
        <th>Country</th>
        <th>Attack type</th>
      </tr>
      {% for a in alerts %}
      <tr>
        <td>{{ a.timestamp }}</td>
        <td>{{ a.user_id }}</td>
        <td>{{ a.ip_address }}</td>
        <td>{{ a.country }}</td>
        <td>{{ a.attack_type }}</td>
      </tr>
      {% endfor %}
    </table>
  {% endif %}
</body>
</html>
"""


@app.route("/")
def index():
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
        alerts = requests.get(f"{API_BASE_URL}/alerts", params={"limit": 100}, timeout=5).json()["alerts"]
        return render_template_string(TEMPLATE, health=health, alerts=alerts, error=None, api_url=API_BASE_URL)
    except requests.exceptions.RequestException as exc:
        return render_template_string(TEMPLATE, error=str(exc), api_url=API_BASE_URL)


if __name__ == "__main__":
    app.run(debug=True)
