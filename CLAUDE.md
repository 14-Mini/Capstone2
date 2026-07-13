# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A capstone project building anomaly/attack detection models on the RBA (Risk-Based Authentication) login dataset. It's a linear script pipeline (no package/module structure, no tests, no build system) that goes: raw 9GB login log → balanced sample → cleaned → feature-engineered → labeled + split → two trained models (Isolation Forest + Random Forest).

There is no requirements.txt/venv checked in. Scripts assume `pandas`, `numpy`, `scikit-learn` are already installed in whatever Python is on PATH. `app.py` (the live serving layer, see below) additionally needs `fastapi`, `pydantic`, and `uvicorn`.

## Running the pipeline

Run scripts directly with `python <script>.py` from the repo root — every script hardcodes its input/output filenames as module-level constants (`INPUT_FILE`, `OUTPUT_FILE`) rather than taking CLI args, and reads/writes CSVs in the current directory.

**Windows gotcha:** several scripts (`clean_data.py`, `label_and_split_v2.py`, `train_isolation_forest.py`) print Unicode arrow characters (`→`). The default Windows terminal encoding (cp1252) can't encode them and the script will crash with `UnicodeEncodeError` partway through. Always run with UTF-8 output forced:
```
PYTHONIOENCODING=utf-8 python <script>.py
```

Full pipeline, in order:
```
PYTHONIOENCODING=utf-8 python sample_balanced_v3.py       # rba-dataset.csv -> rba_sample_16k.csv (~30 min, scans 31M rows twice)
PYTHONIOENCODING=utf-8 python clean_data.py                # -> rba_sample_16k_cleaned.csv
PYTHONIOENCODING=utf-8 python feature_engineering.py       # -> rba_sample_16k_engineered.csv
PYTHONIOENCODING=utf-8 python label_and_split_v2.py        # -> train.csv, val.csv, test.csv, full_labeled.csv
PYTHONIOENCODING=utf-8 python train_isolation_forest.py    # -> isolation_forest.pkl
PYTHONIOENCODING=utf-8 python train_random_forest.py       # -> random_forest.pkl
```
`test.py` is a one-off sanity check (`full_labeled.csv` attack_type distribution) — not part of the pipeline order.

There's no test suite. "Testing" a change means rerunning the affected stage (and everything downstream of it) and comparing the printed metrics/label distribution against a prior run.

## Pipeline architecture

**Sampler → cleaner → feature engineer → labeler/splitter → trainers.** Each stage is a standalone script that reads the previous stage's CSV output and writes its own; there is no shared library code, so behavior changes require editing the constants/logic at the top of the specific script.

### Why the sampler is the tricky stage (read `sample_balanced_v3.py`'s header comment)

`rba-dataset.csv` is 9GB / ~31M rows and is sorted by `Login Timestamp` ascending. The dataset's actual attacks/account-takeovers are extremely rare, and the downstream labeling in `label_and_split_v2.py` depends on *rolling-window* features (`fail_count_1min`, `unique_usernames_5min`, etc. — computed in `feature_engineering.py`) that only become non-zero when multiple close-together events from the same IP/user survive together in the sample.

This went through three sampler generations before landing on `sample_balanced_v3.py` (the earlier `Samplin_data.py`, `sample_balanced.py`, and `sample_balanced_v2.py` were deleted once superseded — history below is preserved here since the files themselves no longer are):
- **v1** (plain uniform/50-50 random row sampling): balanced the raw `Is Attack IP`/`Is Account Takeover` flags but scattered attack bursts across the full file, so almost no rolling-window context survived — the labeled sample ended up ~98% "normal" regardless of the raw balance.
- **v2**: fixed that by picking random attack-IP "seed" events, then pulling every other event from that same IP within a ±10 minute window (mirroring `feature_engineering.py`'s own rolling window) so burst context survives. Introduced two new bugs (see below).
- **v3** (current, `sample_balanced_v3.py`): fixes v2's bugs — (1) real `Is Account Takeover` rows were sharing a capped budget with attack-IP context rows and could get silently dropped before the scan reached them; ATO rows now get their own uncapped bucket. (2) the `classify_attack()` behavioral fallback for account takeover (`geo_anomaly_flag==1 AND time_since_last_login>24h AND success`) almost never fired under v2 sampling because those two conditions were nearly mutually exclusive; v3 additionally captures each ATO user's most-recent prior login (any IP) so `feature_engineering.py` has real data to compute those features from instead of defaulting to "first login ever seen".

The `CONTEXT_WINDOW_MINUTES` constant in the sampler and the rolling-window sizes in `feature_engineering.py` are coupled — if one changes, check the other.

### Labeling is heuristic, not ground-truth (`label_and_split_v2.py`)

`attack_type` (`normal` / `brute_force` / `credential_stuffing` / `dictionary_attack` / `account_takeover`) is derived by `classify_attack()` from hand-tuned thresholds on the engineered rolling-window features, not from a labeled column in the raw data (the raw data only has `Is Attack IP` / `Is Account Takeover` booleans). Priority order matters: account_takeover > credential_stuffing > brute_force > dictionary_attack > normal, checked in that order per-row. `is_anomaly` is just `attack_type != 'normal'`.

The superseded `label_and_split.py` (v1) has been deleted; it used stricter/different thresholds. Thresholds were "revised" (loosened) in v2 specifically because the v1 thresholds were data-driven off percentiles that turned out too strict, leaving too few positive labels. If retuning thresholds, `test.py` (`df['attack_type'].value_counts(normalize=True)`) is the quick way to check the resulting class balance without rerunning the full pipeline.

### Feature set (must stay in sync across 4 files)

The 11 features engineered in `feature_engineering.py` are consumed by `label_and_split_v2.py`'s `classify_attack()`, by `FEATURE_COLS` in both `train_isolation_forest.py` and `train_random_forest.py`, and — for live inference — recomputed online by `app.py`'s `SessionStore.compute_features()`:
```
time_of_day, ip_reputation_score, geo_anomaly_flag, fail_count_1min, fail_count_5min,
unique_usernames_5min, rolling_fail_velocity, login_success_rate_1min,
time_since_last_login, fail_to_success_ratio_5min, is_new_device
```
`is_new_device` is also read directly by `classify_attack()`'s behavioral account-takeover rule (geo anomaly + long time gap + success + new device), not just fed to the models as an input. If you add/rename/remove a feature in `feature_engineering.py`, update `FEATURE_COLS` in both training scripts, any threshold logic in `classify_attack()` that reads it, and `compute_features()` in `app.py` — a mismatch there won't fail at training time, it'll surface as a `KeyError` the first time `/predict` is called after retraining.

Both training scripts explicitly exclude `is_account_takeover, is_attack_ip, attack_type, is_anomaly` from `FEATURE_COLS` — these are ground-truth/target columns and including them would be data leakage.

### The two models solve different problems

- **`train_isolation_forest.py`**: unsupervised, trained *only on rows labeled `normal`* in the training split, predicts binary anomaly vs `is_anomaly` ground truth on validation. Sweeps `contamination` over a fixed grid (`[0.05, 0.08, 0.10, 0.13, 0.15, 0.20, 0.25]`) and keeps whichever gives the best F1 — the printed "best contamination" isn't a fixed constant, it's chosen per run based on the current validation split.
- **`train_random_forest.py`**: supervised multiclass on `attack_type` directly (all 5 classes, not just normal/anomaly), `class_weight='balanced'`. Because `account_takeover` is a tiny minority class even after the v3 sampling fix, expect its precision to be much worse than the other classes' (recall is usually fine, precision suffers from false positives pulled from the `normal` class) — this is an inherent class-imbalance property of the data, not a bug to "fix" by touching the sampler again.

Both training scripts save a dict (not a bare model) to their `.pkl`: `{'model', 'feature_cols', ...metadata}`, and self-verify after saving by reloading and comparing predictions on a validation slice — if you change the save format, keep that self-check in sync.

## Serving layer (`app.py`)

A FastAPI app that serves both trained models for live inference, recomputing the same features `feature_engineering.py` computes offline — but online, per-request, instead of from a static CSV. Run it with `uvicorn app:app --reload`.

**State:** a single in-process `SessionStore` (no database) holds three scopes, seeded once at startup from `full_labeled.csv` via `seed_from_csv()` (called from the FastAPI `startup` event) and then updated on every `/predict` call:
- `ip_history` — per-IP rolling list of `(timestamp, user_id, success)`, pruned to the last 10 minutes on each access. Feeds `fail_count_1min/5min`, `unique_usernames_5min`, `rolling_fail_velocity`, `login_success_rate_1min`, `fail_to_success_ratio_5min`.
- `user_last_login` / `user_country_counts` / `user_devices` — per-user profile, persists indefinitely (unwindowed). Feeds `time_since_last_login`, `geo_anomaly_flag`, `is_new_device`.
- `ip_reputation` — static lookup, each IP's historical mean of `is_attack_ip` computed once from the seed CSV at startup; never updated afterward, so an IP with no seed history scores reputation 0.0 regardless of what it does live.

State is in-memory and process-local: restarting the app forgets everything learned from live `/predict` traffic and re-seeds from `full_labeled.csv` only.

**Endpoints:**
- `POST /predict` — takes a `LoginEvent`, computes features via `SessionStore.compute_features()`, runs both models (RF → `attack_type` + per-class probabilities, IF → `is_anomaly` + `anomaly_score`), records an alert if `is_anomaly` or `attack_type != "normal"`, then ingests the event into the store — in that order, so a request's own event never contaminates its own features.
- `GET /alerts?limit=100` — the live alert feed: `SessionStore.alerts` is a `deque(maxlen=500)`, newest-first (`appendleft`, so it prunes oldest at capacity). Each entry also carries the full `features` dict from that request (needed for the dashboard's fail-velocity trend). An event is recorded whenever `is_anomaly` OR `attack_type != "normal"` — the two conditions come from different models and don't always agree; a request can show up as `attack_type: normal` with `is_anomaly: true` when IF flags it before RF's classification does. That's expected, not a bug.

## Dashboard (`dashboard.py`)

Streamlit dashboard for the 12-week plan's Week 11 deliverable, polling `app.py`'s `/alerts` and `/health` — no direct DB/model access of its own except loading `isolation_forest.pkl` once (`@st.cache_resource`) to read `offset_` for the gauge's decision-boundary line. Run with `streamlit run dashboard.py` (needs `streamlit`, `plotly`, `pycountry` — not required by the batch pipeline or `app.py`). `API_BASE_URL` at the top of the file defaults to `http://127.0.0.1:8000`.

Panels: KPI row (from `/health`), an anomaly-score gauge with the IF decision boundary marked, an attack-type distribution bar (fixed color per class), a country choropleth (alpha-2 → alpha-3 via `pycountry`, since the RBA dataset's `country` field is alpha-2), a `rolling_fail_velocity` trend line, and the raw alert table. Auto-refreshes via `time.sleep()` + `st.rerun()` at the bottom of the script (no external autorefresh package). Handles the zero-alerts state explicitly (`st.info` instead of rendering empty charts) and shows an error with a stop if the API is unreachable.

Verified in a real browser via Playwright (`chromium.launch()` — no `chromium-cli` in this environment): both the populated state (seeded brute-force/credential-stuffing/normal events across DE/FR/US) and the empty state render cleanly with no traceback and no console errors. Note the first paint is slow (~10s, Plotly's JS bundle loading inside Streamlit) — don't mistake the gray skeleton placeholders for a failure if you screenshot too early.
- `GET /session/{user_id}` — debug endpoint: a user's tracked state, plus an IP's rolling history/reputation if `ip_address` is passed as a query param.
- `GET /health` — model-loaded flag and store size counters, including `alerts_in_feed`.

Smoke-tested in `fastapi_test_report.png`: 8 simulated failed logins from one IP against distinct usernames, 1s apart, sent to the live endpoint. The IF anomaly score crosses the decision boundary at request 3; the RF classification correctly escalates `normal` → `brute_force` → `credential_stuffing` as `unique_usernames_5min` climbs past `classify_attack()`'s credential-stuffing threshold — expected (credential stuffing is checked before brute force and is essentially "brute force against many distinct usernames from one source"), not a bug.

**Known limitation — profile poisoning (found via `test_account_takeover_smoke.py`, `fastapi_ato_smoke_report.png`):** `user_country_counts` and `user_devices` are unbounded, unwindowed, and updated by `ingest()` on *every* request regardless of whether that request was flagged anomalous. In a sweep that replayed the same simulated attacker (new country + new device, successful logins) against one seeded user at increasing time gaps, `geo_anomaly_flag` correctly fired for the first 7 requests, then dropped to 0 permanently — the attacker's own country had accumulated enough logins in `user_country_counts` to become `most_common()`, so the store started treating the attacker's country as the user's legitimate baseline. `is_new_device` has the same failure mode (fires once, then the attacker's device is "known"). Net effect: this rule only reliably catches the *first* login of a sustained takeover — a repeat attacker from the same country/device blends into the profile within a handful of requests. RF's `account_takeover` probability did respond to the geo/device/time-gap combination while it lasted (0.32–0.58 across the sweep, occasionally winning the argmax) but never distinguished gaps above vs. below the 24h `ATO_TIME_GAP` label threshold the way the offline rule does, and IF's anomaly score stayed well under its own decision boundary for every point in this scenario. This is a gap in the online feature computation, not something `feature_engineering.py`'s one-shot batch computation over a fixed CSV would exhibit the same way. Dictionary-attack path is still untested live.
