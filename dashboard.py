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
  <title>Failed Attack Detection</title>
  <style>
    body {
      font-family: sans-serif;
      margin: 2rem;
      background: #f7f8fa;
      color: #1a1a1a;
    }
    h1 {
      font-size: 1.4rem;
      margin-bottom: 1.2rem;
    }
    .kpi {
      display: flex;
      gap: 1rem;
      margin-bottom: 1.5rem;
    }
    .kpi div {
      background: #fff;
      border: 1px solid #e0e0e0;
      border-radius: 6px;
      padding: 0.75rem 1.25rem;
      min-width: 100px;
    }
    .kpi .label {
      display: block;
      font-size: 0.75rem;
      color: #777;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .kpi .value {
      display: block;
      font-size: 1.5rem;
      font-weight: bold;
      margin-top: 2px;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      background: #fff;
      border-radius: 6px;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    th, td {
      padding: 8px 12px;
      text-align: left;
      font-size: 0.9rem;
    }
    th {
      background: #eceff3;
      font-weight: 600;
    }
    tbody tr:nth-child(even) {
      background: #fafbfc;
    }
    tbody tr:hover {
      background: #f0f4ff;
    }
    .attack-type {
      font-weight: 600;
    }
    .attack-type.normal {
      color: #2e7d32;
    }
    .attack-type.alert {
      color: #c62828;
    }
    .login-result.success {
      color: #2e7d32;
      font-weight: 600;
    }
    .login-result.failed {
      color: #c62828;
      font-weight: 600;
    }
  </style>
</head>
<body>
  <h1>Failed Login Attack Detection &mdash; Live Dashboard</h1>

  {% if error %}
    <p style="color:#c62828">Cannot reach the API at {{ api_url }}: {{ error }}</p>
  {% else %}
    <div class="kpi">
      <div><span class="label">Alerts</span><span class="value">{{ health.alerts_in_feed }}</span></div>
      <div><span class="label">Users</span><span class="value">{{ health.users_tracked }}</span></div>
      <div><span class="label">IPs</span><span class="value">{{ health.ips_tracked }}</span></div>
      <div><span class="label">Models loaded</span><span class="value">{{ "yes" if health.models_loaded else "no" }}</span></div>
    </div>
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>User</th>
          <th>IP</th>
          <th>Country</th>
          <th>Device</th>
          <th>Login</th>
          <th>Attack type</th>
        </tr>
      </thead>
      <tbody>
        {% for a in alerts %}
        <tr>
          <td>{{ a.timestamp }}</td>
          <td>{{ a.user_id }}</td>
          <td>{{ a.ip_address }}</td>
          <td>{{ a.country }}</td>
          <td>{{ a.device_type }}</td>
          <td class="login-result {{ 'success' if a.login_successful else 'failed' }}">{{ 'Success' if a.login_successful else 'Failed' }}</td>
          <td class="attack-type {{ 'normal' if a.attack_type == 'normal' else 'alert' }}">{{ a.attack_type }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  {% endif %}
</body>
</html>
"""


@app.route("/")
def index():
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        health = health_response.json()

        alerts_response = requests.get(f"{API_BASE_URL}/alerts", params={"limit": 100}, timeout=5)
        alerts = alerts_response.json()["alerts"]

        return render_template_string(TEMPLATE, health=health, alerts=alerts, error=None, api_url=API_BASE_URL)
    except requests.exceptions.RequestException as exc:
        return render_template_string(TEMPLATE, error=str(exc), api_url=API_BASE_URL)


if __name__ == "__main__":
    app.run(debug=True)
