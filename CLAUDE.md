# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A capstone project building anomaly/attack detection models on the RBA (Risk-Based Authentication) login dataset. It's a linear script pipeline (no package/module structure, no build system) that goes: raw 9GB login log → balanced sample → cleaned → feature-engineered → labeled + split → two trained models (Isolation Forest + Random Forest).

`requirements.txt` pins every dependency across the pipeline, `app.py`, and `dashboard.py` (versions match what this repo has been developed/tested against). No venv is checked in.

## Layout

```
app.py, dashboard.py, CLAUDE.md, requirements.txt, render.yaml   # deployed entry points + docs, stay at repo root
scripts/      # pipeline stages (sampler, cleaner, feature engineer, labeler/splitter, trainers)
tests/        # test.py sanity check, test_account_takeover_smoke.py live smoke test
data/raw/     # rba-dataset.csv (9GB, gitignored, not checked in)
data/interim/ # rba_sample_16k.csv and its cleaned/engineered derivatives
data/splits/  # train.csv, val.csv, test.csv, full_labeled.csv
models/       # isolation_forest.pkl, random_forest.pkl
reports/      # classification reports, ROC/AUC plots, smoke-test output
docs/         # project proposal, presentation slides
```

## Running the pipeline

Run scripts as `python scripts/<script>.py` from the repo root — every script hardcodes its input/output paths as module-level constants (`INPUT_FILE`, `OUTPUT_FILE`) rather than taking CLI args. Those paths are relative to the repo root (`data/...`, `models/...`), so always invoke from the repo root, not from inside `scripts/`.

**Windows gotcha:** several scripts (`clean_data.py`, `label_and_split_v2.py`, `train_isolation_forest.py`) print Unicode arrow characters (`→`). The default Windows terminal encoding (cp1252) can't encode them and the script will crash with `UnicodeEncodeError` partway through. Always run with UTF-8 output forced:
```
PYTHONIOENCODING=utf-8 python scripts/<script>.py
```

Full pipeline, in order:
```
PYTHONIOENCODING=utf-8 python scripts/sample_balanced_v3.py       # data/raw/rba-dataset.csv -> data/interim/rba_sample_16k.csv (~30 min, scans 31M rows twice)
PYTHONIOENCODING=utf-8 python scripts/clean_data.py                # -> data/interim/rba_sample_16k_cleaned.csv
PYTHONIOENCODING=utf-8 python scripts/feature_engineering.py       # -> data/interim/rba_sample_16k_engineered.csv
PYTHONIOENCODING=utf-8 python scripts/label_and_split_v2.py        # -> data/splits/{train,val,test,full_labeled}.csv
PYTHONIOENCODING=utf-8 python scripts/train_isolation_forest.py    # -> models/isolation_forest.pkl
PYTHONIOENCODING=utf-8 python scripts/train_random_forest.py       # -> models/random_forest.pkl
```
`tests/test.py` is a one-off sanity check (`data/splits/full_labeled.csv` attack_type distribution) — not part of the pipeline order.

There's no test suite. "Testing" a change means rerunning the affected stage (and everything downstream of it) and comparing the printed metrics/label distribution against a prior run.

## Pipeline architecture

**Sampler → cleaner → feature engineer → labeler/splitter → trainers.** Each stage is a standalone script that reads the previous stage's CSV output and writes its own; there is no shared library code, so behavior changes require editing the constants/logic at the top of the specific script.

### Why the sampler is the tricky stage (read `scripts/sample_balanced_v3.py`'s header comment)

`data/raw/rba-dataset.csv` is 9GB / ~31M rows and is sorted by `Login Timestamp` ascending. The dataset's actual attacks/account-takeovers are extremely rare, and the downstream labeling in `label_and_split_v2.py` depends on *rolling-window* features (`fail_count_1min`, `unique_usernames_5min`, etc. — computed in `feature_engineering.py`) that only become non-zero when multiple close-together events from the same IP/user survive together in the sample.

This went through three sampler generations before landing on `scripts/sample_balanced_v3.py` (the earlier `Samplin_data.py`, `sample_balanced.py`, and `sample_balanced_v2.py` were deleted once superseded — history below is preserved here since the files themselves no longer are):
- **v1** (plain uniform/50-50 random row sampling): balanced the raw `Is Attack IP`/`Is Account Takeover` flags but scattered attack bursts across the full file, so almost no rolling-window context survived — the labeled sample ended up ~98% "normal" regardless of the raw balance.
- **v2**: fixed that by picking random attack-IP "seed" events, then pulling every other event from that same IP within a ±10 minute window (mirroring `feature_engineering.py`'s own rolling window) so burst context survives. Introduced two new bugs (see below).
- **v3** (current, `scripts/sample_balanced_v3.py`): fixes v2's bugs — (1) real `Is Account Takeover` rows were sharing a capped budget with attack-IP context rows and could get silently dropped before the scan reached them; ATO rows now get their own uncapped bucket. (2) the `classify_attack()` behavioral fallback for account takeover (`geo_anomaly_flag==1 AND time_since_last_login>24h AND success`) almost never fired under v2 sampling because those two conditions were nearly mutually exclusive; v3 additionally captures each ATO user's most-recent prior login (any IP) so `feature_engineering.py` has real data to compute those features from instead of defaulting to "first login ever seen".

The `CONTEXT_WINDOW_MINUTES` constant in the sampler and the rolling-window sizes in `scripts/feature_engineering.py` are coupled — if one changes, check the other.

### Labeling is heuristic, not ground-truth (`scripts/label_and_split_v2.py`)

`attack_type` (`normal` / `brute_force` / `credential_stuffing` / `dictionary_attack` / `account_takeover`) is derived by `classify_attack()` from hand-tuned thresholds on the engineered rolling-window features, not from a labeled column in the raw data (the raw data only has `Is Attack IP` / `Is Account Takeover` booleans). Priority order matters: account_takeover > credential_stuffing > brute_force > dictionary_attack > normal, checked in that order per-row. `is_anomaly` is just `attack_type != 'normal'`.

The superseded `label_and_split.py` (v1) has been deleted; it used stricter/different thresholds. Thresholds were "revised" (loosened) in v2 specifically because the v1 thresholds were data-driven off percentiles that turned out too strict, leaving too few positive labels. If retuning thresholds, `tests/test.py` (`df['attack_type'].value_counts(normalize=True)`) is the quick way to check the resulting class balance without rerunning the full pipeline.

### Feature set (must stay in sync across 4 files)

The 11 features engineered in `scripts/feature_engineering.py` are consumed by `scripts/label_and_split_v2.py`'s `classify_attack()`, by `FEATURE_COLS` in both `scripts/train_isolation_forest.py` and `scripts/train_random_forest.py`, and — for live inference — recomputed online by `app.py`'s `SessionStore.compute_features()`:
```
time_of_day, ip_reputation_score, geo_anomaly_flag, fail_count_1min, fail_count_5min,
unique_usernames_5min, rolling_fail_velocity, login_success_rate_1min,
time_since_last_login, fail_to_success_ratio_5min, is_new_device
```
`is_new_device` is also read directly by `classify_attack()`'s behavioral account-takeover rule (geo anomaly + long time gap + success + new device), not just fed to the models as an input. If you add/rename/remove a feature in `scripts/feature_engineering.py`, update `FEATURE_COLS` in both training scripts, any threshold logic in `classify_attack()` that reads it, and `compute_features()` in `app.py` — a mismatch there won't fail at training time, it'll surface as a `KeyError` the first time `/predict` is called after retraining.

Both training scripts explicitly exclude `is_account_takeover, is_attack_ip, attack_type, is_anomaly` from `FEATURE_COLS` — these are ground-truth/target columns and including them would be data leakage.

### The two models solve different problems (but only one is served live)

- **`scripts/train_isolation_forest.py`**: unsupervised, trained *only on rows labeled `normal`* in the training split, predicts binary anomaly vs `is_anomaly` ground truth on validation. Sweeps `contamination` over a fixed grid (`[0.05, 0.08, 0.10, 0.13, 0.15, 0.20, 0.25]`) and keeps whichever gives the best F1 — the printed "best contamination" isn't a fixed constant, it's chosen per run based on the current validation split. Still trained and saved to `models/isolation_forest.pkl` as part of the pipeline, but as of the `app.py` simplification below, **it's no longer loaded or served live** — this model exists offline only now.
- **`scripts/train_random_forest.py`**: supervised multiclass on `attack_type` directly (all 5 classes, not just normal/anomaly), `class_weight='balanced'`. Because `account_takeover` is a tiny minority class even after the v3 sampling fix, expect its precision to be much worse than the other classes' (recall is usually fine, precision suffers from false positives pulled from the `normal` class) — this is an inherent class-imbalance property of the data, not a bug to "fix" by touching the sampler again. This is the only model `app.py` serves.

Both training scripts save a dict (not a bare model) to their `.pkl`: `{'model', 'feature_cols', ...metadata}`, and self-verify after saving by reloading and comparing predictions on a validation slice — if you change the save format, keep that self-check in sync.

## Serving layer (`app.py`)

A small FastAPI app (~180 lines, deliberately kept minimal) that serves the Random Forest model for live inference, recomputing the same 11 features `feature_engineering.py` computes offline — but online, per-request, instead of from a static CSV. Run it with `uvicorn app:app --reload`.

**State:** a single in-process `SessionStore` (no database) holds three scopes, all built purely from live `/predict` traffic — there is no startup seeding from a CSV:
- `ip_history` — per-IP rolling list of `(timestamp, user_id, success)`, pruned to the last 10 minutes on each access. Feeds `fail_count_1min/5min`, `unique_usernames_5min`, `rolling_fail_velocity`, `login_success_rate_1min`, `fail_to_success_ratio_5min`.
- `user_last_login` / `user_country_counts` / `user_devices` — per-user profile, persists indefinitely (unwindowed). Feeds `time_since_last_login`, `geo_anomaly_flag`, `is_new_device`.
- `ip_reputation` — kept as an empty dict for feature-shape compatibility (the model still expects an `ip_reputation_score` column), but nothing ever populates it, so this feature reads `0.0` for all live traffic. Not a bug — just an accepted simplification since nothing currently feeds it real reputation data.

State is in-memory and process-local: restarting the app forgets everything learned from live `/predict` traffic and starts completely empty.

**Endpoints:**
- `POST /predict` — takes a `LoginEvent`, computes features via `SessionStore.compute_features()`, runs the Random Forest (→ `attack_type` + per-class probabilities), records an alert whenever `attack_type != "normal"`, then ingests the event into the store — in that order, so a request's own event never contaminates its own features.
- `GET /alerts?limit=100` — the live alert feed: `SessionStore.alerts` is a `deque(maxlen=500)`, newest-first (`appendleft`, so it prunes oldest at capacity). Each entry also carries the full `features` dict from that request. Every entry is non-normal by definition (that's the only condition that puts something in the feed) — there's no separate `is_anomaly` flag or anomaly score anymore.
- `GET /health` — model-loaded flag and store size counters, including `alerts_in_feed`.

There is no debug endpoint for inspecting a single user's/IP's tracked state (the old `GET /session/{user_id}` was dropped as unused).

## Dashboard (`dashboard.py`)

A minimal Flask app, deliberately kept small (one route, an inline HTML template with a little CSS, no charting/JS): a KPI row (alerts in feed, users tracked, IPs tracked, models-loaded flag, from `/health`) and the raw live alert table (timestamp, user, IP, country, attack type — from `/alerts`), rendered with `flask.render_template_string`. No direct DB/model access of its own — everything comes from `app.py`'s `/health` and `/alerts`. Run with `python dashboard.py` (needs only `flask` and `requests` beyond the pipeline's deps); serves on `http://127.0.0.1:5000` by default. Refresh is manual — reload the page in the browser; there's no polling loop or meta-refresh.

`API_BASE_URL` resolves from the `API_BASE_URL` env var, falling back to `http://127.0.0.1:8000` for local dev. If the backend is unreachable, the `/` route catches `requests.exceptions.RequestException` and renders an inline error message instead of a stack trace.

## Login page (`login.py`)

A separate, minimal Flask app — a plain white username/password form, unrelated in purpose to `dashboard.py`. Checks credentials against `data/users.db` (SQLite, seeded by `scripts/seed_users.py`, plaintext passwords — deliberately simplified for this demo, not a real auth pattern). Runs on `http://127.0.0.1:5001` by default (`python login.py`) so it doesn't collide with `dashboard.py` (5000) or `app.py`/uvicorn (8000).

On every submitted attempt, after deciding the login success/fail message from the DB check, `login.py` also POSTs the event to `app.py`'s `POST /predict` — fire-and-forget, wrapped in a `try/except requests.exceptions.RequestException: pass`, so an unreachable backend never affects the login page itself (monitoring only, not a gate: the model's verdict never blocks or changes a login outcome). `APP_BASE_URL` resolves the same way `dashboard.py`'s `API_BASE_URL` does (env var, falling back to `http://127.0.0.1:8000`).

Because a 2-field login form doesn't naturally have `country`/`device_type`/`browser_name_and_version`/`os_name_and_version` (all required by `LoginEvent`), these are filled in as best-effort: `ip_address` is the real connecting IP (`request.remote_addr`), device/browser/OS are parsed from the `User-Agent` header via `parse_user_agent()` (simple substring checks, no new dependency), and `country` is a fixed placeholder (`"US"`) since there's no GeoIP source wired up — meaning `geo_anomaly_flag` won't meaningfully fire from traffic through this page. Documented limitation, not a bug.

## Deployment

Two services, deployed separately:

- **Backend (`app.py`)**: Render, via `render.yaml` (Blueprint) or a manual Web Service with build command `pip install -r requirements.txt` and start command `uvicorn app:app --host 0.0.0.0 --port $PORT`. No CORS setup needed — the dashboard calls it server-to-server via `requests`, not from the browser.
- **Dashboard (`dashboard.py`)**: a plain Flask app, so it can run anywhere Python runs — locally (`python dashboard.py`), or as its own small Render/Flask web service (build command `pip install -r requirements.txt`, start command `gunicorn dashboard:app` or similar). Point it at the backend by setting the `API_BASE_URL` env var, e.g. `https://<your-render-service>.onrender.com`.
- Render's free tier spins down on idle and cold-starts slowly (~50s) — expect the dashboard's first load after backend inactivity to show the "cannot reach the API" error briefly before Render wakes the backend up.
- The backend reads `models/random_forest.pkl` straight from the repo checkout at startup — nothing else to provision (no database, no object storage). `models/isolation_forest.pkl` still exists in the repo as a trained artifact but isn't read by the backend anymore.

Smoke-tested in `reports/fastapi_test_report.png`: 8 simulated failed logins from one IP against distinct usernames, 1s apart, sent to the live endpoint. The RF classification correctly escalates `normal` → `brute_force` → `credential_stuffing` as `unique_usernames_5min` climbs past `classify_attack()`'s credential-stuffing threshold — expected (credential stuffing is checked before brute force and is essentially "brute force against many distinct usernames from one source"), not a bug. (This test predates the Isolation Forest being dropped from serving; the RF escalation behavior it demonstrates is unaffected by that change.)

**Known limitation — profile poisoning (found via `tests/test_account_takeover_smoke.py`, `reports/fastapi_ato_smoke_report.png`):** `user_country_counts` and `user_devices` are unbounded, unwindowed, and updated by `ingest()` on *every* request regardless of whether that request was flagged anomalous. In a sweep that replayed the same simulated attacker (new country + new device, successful logins) against one seeded user at increasing time gaps, `geo_anomaly_flag` correctly fired for the first 7 requests, then dropped to 0 permanently — the attacker's own country had accumulated enough logins in `user_country_counts` to become `most_common()`, so the store started treating the attacker's country as the user's legitimate baseline. `is_new_device` has the same failure mode (fires once, then the attacker's device is "known"). Net effect: this rule only reliably catches the *first* login of a sustained takeover — a repeat attacker from the same country/device blends into the profile within a handful of requests. RF's `account_takeover` probability did respond to the geo/device/time-gap combination while it lasted (0.32–0.58 across the sweep, occasionally winning the argmax) but never distinguished gaps above vs. below the 24h `ATO_TIME_GAP` label threshold the way the offline rule does. This is a gap in the online feature computation, not something `feature_engineering.py`'s one-shot batch computation over a fixed CSV would exhibit the same way. Dictionary-attack path is still untested live.
