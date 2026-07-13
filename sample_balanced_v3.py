import pandas as pd
import numpy as np
import time
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "rba-dataset.csv"
OUTPUT_FILE = "rba_sample_16k.csv"
CHUNK_SIZE = 500000
RANDOM_SEED = 42

NORMAL_BUDGET = 8000
ANOMALY_BUDGET = 8000          # attack-IP context rows only (account-takeover is separate, see below)

SEED_COUNT = 4000              # number of attack-IP events used as context anchors
CONTEXT_WINDOW_MINUTES = 10    # matches the 10-min rolling window in feature_engineering.py
MAX_NEIGHBORS_PER_SEED_IP = 15  # cap so one bursty IP doesn't eat the whole anomaly budget
WINDOW_SECONDS = CONTEXT_WINDOW_MINUTES * 60

# ============================================================================
# WHAT'S NEW IN v3 (vs sample_balanced_v2.py)
# ============================================================================
# v2 had two account-takeover problems, found by inspecting the labeled output:
#
# 1. BUG: real "Is Account Takeover" rows were added to the same `context_rows`
#    bucket as attack-IP neighbors, capped by the shared ANOMALY_BUDGET. If that
#    budget filled up from attack-IP context before the scan reached a true
#    account-takeover row later in the 31M-row file, that row was silently
#    dropped. Account takeovers are the rarest, highest-value class -- they now
#    get their own uncapped bucket, guaranteeing every real one in the file is
#    kept.
#
# 2. The behavioral fallback in label_and_split_v2.py's classify_attack()
#    (geo_anomaly_flag==1 AND time_since_last_login>24h AND success) never
#    fired: geo_anomaly_flag can only be 1 if a user appears at least twice in
#    the sample with different countries, but time_since_last_login is only
#    large if that user's *previous* login is missing from the sample -- these
#    two conditions are close to mutually exclusive under v2's sampling. So
#    v3 additionally captures, for every real account-takeover event, that
#    user's most recent prior login (any IP) so feature_engineering.py has
#    something real to compute geo_anomaly_flag / time_since_last_login from
#    instead of defaulting to "first login ever seen".
#
# This relies on the raw file being sorted by Login Timestamp ascending
# (verified true for this dataset), so a single forward pass naturally
# encounters a user's earlier logins before their later ones.
# ============================================================================

print("=" * 60)
print("RBA Balanced Sampler v3 (+ guaranteed account-takeover context)")
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Target: {ANOMALY_BUDGET:,} anomaly-context + {NORMAL_BUDGET:,} normal rows "
      f"+ all account-takeover rows (uncapped)")
print("=" * 60)

start_time = time.time()
rng = np.random.default_rng(RANDOM_SEED)

# ============================================================================
# PASS 1: reservoir-sample seed attack-IP events + collect all ATO (user, ts)
# ============================================================================

print("\n[Pass 1/2] Sampling seed attack-IP events + locating account-takeover events...")

seed_count = 0
seed_records = []  # list of (ip_str, ts_seconds)
ato_events = []    # list of (user_id_str, ts_seconds) -- expected to be tiny
rows_scanned = 0

for chunk in pd.read_csv(INPUT_FILE, usecols=['IP Address', 'User ID', 'Login Timestamp',
                                                'Is Attack IP', 'Is Account Takeover'],
                          chunksize=CHUNK_SIZE, encoding='utf-8', low_memory=False):
    is_attack = chunk['Is Attack IP'].astype(str).str.strip().str.lower() == 'true'
    is_ato = chunk['Is Account Takeover'].astype(str).str.strip().str.lower() == 'true'

    seed_chunk = chunk[is_attack]
    if len(seed_chunk):
        ts_seconds = pd.to_datetime(seed_chunk['Login Timestamp'], errors='coerce').astype('int64') // 10**9
        ips = seed_chunk['IP Address'].astype(str)
        for ip, t in zip(ips.tolist(), ts_seconds.tolist()):
            seed_count += 1
            if len(seed_records) < SEED_COUNT:
                seed_records.append((ip, int(t)))
            else:
                j = rng.integers(0, seed_count)
                if j < SEED_COUNT:
                    seed_records[j] = (ip, int(t))

    ato_chunk = chunk[is_ato]
    if len(ato_chunk):
        ato_ts = pd.to_datetime(ato_chunk['Login Timestamp'], errors='coerce').astype('int64') // 10**9
        ato_users = ato_chunk['User ID'].astype(str)
        for u, t in zip(ato_users.tolist(), ato_ts.tolist()):
            ato_events.append((u, int(t)))

    rows_scanned += len(chunk)
    print(f"  scanned {rows_scanned:,} rows | seed reservoir {len(seed_records):,}/{SEED_COUNT:,} | "
          f"ATO events found so far: {len(ato_events):,}", end='\r', flush=True)

print()

seed_ip_to_ts = defaultdict(list)
for ip, t in seed_records:
    seed_ip_to_ts[ip].append(t)
seed_ip_set = set(seed_ip_to_ts.keys())

ato_user_to_ts = defaultdict(list)
for u, t in ato_events:
    ato_user_to_ts[u].append(t)
ato_user_set = set(ato_user_to_ts.keys())

print(f"  Seed events: {len(seed_records):,} across {len(seed_ip_set):,} unique IPs")
print(f"  Account-takeover events: {len(ato_events):,} across {len(ato_user_set):,} unique users")

# ============================================================================
# PASS 2: expand seeds/ATO context + sample normal rows
# ============================================================================

print("\n[Pass 2/2] Expanding context windows and sampling normal rows...")

context_rows = {}       # global_row_idx -> row dict (attack-IP context, capped)
context_count_per_ip = defaultdict(int)

ato_rows = {}            # global_row_idx -> row dict (real ATO rows, uncapped)
ato_prior_login = {}     # user_id -> row dict (most recent login seen so far, before their ATO event)
ato_context_rows = {}    # global_row_idx -> row dict (each ATO user's prior login, uncapped)

normal_reservoir = []
normal_seen = 0
rows_scanned = 0
global_idx = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, encoding='utf-8', low_memory=False):
    chunk_ip = chunk['IP Address'].astype(str)
    chunk_user = chunk['User ID'].astype(str)
    ts_seconds = pd.to_datetime(chunk['Login Timestamp'], errors='coerce').astype('int64') // 10**9
    is_ato = chunk['Is Account Takeover'].astype(str).str.strip().str.lower() == 'true'
    is_seed_ip = chunk_ip.isin(seed_ip_set)
    is_ato_user = chunk_user.isin(ato_user_set)
    records = chunk.to_dict('records')

    for i in range(len(chunk)):
        row_global_idx = global_idx + i
        added = False

        # Track each ATO user's most recent login BEFORE their takeover event
        # (file is timestamp-sorted, so the last update before we hit the ATO
        # row itself is the closest prior session).
        if is_ato_user.iat[i] and not is_ato.iat[i]:
            user = chunk_user.iat[i]
            t = int(ts_seconds.iat[i])
            if any(t < ato_t for ato_t in ato_user_to_ts[user]):
                ato_prior_login[user] = (row_global_idx, records[i])

        if is_ato.iat[i]:
            ato_rows[row_global_idx] = records[i]
            user = chunk_user.iat[i]
            if user in ato_prior_login:
                prior_idx, prior_row = ato_prior_login[user]
                ato_context_rows[prior_idx] = prior_row
            added = True
        elif is_seed_ip.iat[i] and len(context_rows) < ANOMALY_BUDGET:
            ip = chunk_ip.iat[i]
            if context_count_per_ip[ip] < MAX_NEIGHBORS_PER_SEED_IP:
                t = int(ts_seconds.iat[i])
                if any(abs(t - st) <= WINDOW_SECONDS for st in seed_ip_to_ts[ip]):
                    context_rows[row_global_idx] = records[i]
                    context_count_per_ip[ip] += 1
                    added = True

        if not added:
            normal_seen += 1
            if len(normal_reservoir) < NORMAL_BUDGET:
                normal_reservoir.append(records[i])
            else:
                j = rng.integers(0, normal_seen)
                if j < NORMAL_BUDGET:
                    normal_reservoir[j] = records[i]

    global_idx += len(chunk)
    rows_scanned += len(chunk)
    print(f"  scanned {rows_scanned:,} rows | context {len(context_rows):,}/{ANOMALY_BUDGET:,} | "
          f"ATO {len(ato_rows):,} (+{len(ato_context_rows):,} prior-login context) | "
          f"normal {len(normal_reservoir):,}/{NORMAL_BUDGET:,}",
          end='\r', flush=True)

print()

if len(context_rows) < ANOMALY_BUDGET:
    print(f"\nWARNING: only found {len(context_rows):,} anomaly-context rows "
          f"(wanted {ANOMALY_BUDGET:,}). Consider raising SEED_COUNT or MAX_NEIGHBORS_PER_SEED_IP.")

missing_prior = len(ato_rows) - len(ato_context_rows)
if missing_prior > 0:
    print(f"\nNOTE: {missing_prior:,} account-takeover events had no earlier login "
          f"for that user anywhere in the file (first-ever login) -- no prior-login "
          f"context row to add for those.")

# ============================================================================
# COMBINE, DEDUPE, SHUFFLE, SAVE
# ============================================================================

print("\nCombining, deduping and shuffling...")

# ato_context_rows (prior logins) could overlap with context_rows/normal_reservoir
# global indices in theory; dict keys make ato/context dedup automatic. Drop any
# ato_context_rows key that's already an ato_row (shouldn't happen, but be safe).
for idx in list(ato_context_rows.keys()):
    if idx in ato_rows:
        del ato_context_rows[idx]

df_context = pd.DataFrame(list(context_rows.values()))
df_ato = pd.DataFrame(list(ato_rows.values()))
df_ato_prior = pd.DataFrame(list(ato_context_rows.values()))
df_normal = pd.DataFrame(normal_reservoir)

df_sample = pd.concat([df_context, df_ato, df_ato_prior, df_normal], ignore_index=True)
df_sample = df_sample.drop_duplicates()
df_sample = df_sample.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

df_sample.to_csv(OUTPUT_FILE, index=False)

elapsed = time.time() - start_time

print(f"\n{'=' * 60}")
print("DONE")
print(f"Rows scanned:            {rows_scanned:,}")
print(f"Seed attack-IP events:    {len(seed_records):,} across {len(seed_ip_set):,} IPs")
print(f"Anomaly-context rows:     {len(context_rows):,}")
print(f"Account-takeover rows:    {len(ato_rows):,}")
print(f"ATO prior-login context:  {len(ato_context_rows):,}")
print(f"Normal rows:              {len(normal_reservoir):,}")
print(f"Output rows:              {len(df_sample):,}")
print(f"Output: {OUTPUT_FILE}")
print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"{'=' * 60}")
