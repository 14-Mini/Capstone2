import pandas as pd
import numpy as np
import re
import time
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "rba_sample_16k_cleaned.csv"
OUTPUT_FILE = "rba_sample_16k_engineered.csv"

# Time windows for rolling features (in minutes)
WINDOW_1MIN = 1
WINDOW_5MIN = 5
WINDOW_10MIN = 10

# Sentinel value for "no previous login" (in seconds)
NO_PREVIOUS_LOGIN = 999999

print("=" * 60)
print("Feature Engineering - RBA Sample (11 Features)")
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print("=" * 60)

start_time = time.time()

# ============================================================================
# STEP 1: LOAD CLEANED DATA
# ============================================================================

print("\n[1/7] Loading cleaned data...")

try:
    df = pd.read_csv(INPUT_FILE, encoding='utf-8', low_memory=False)
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"  ERROR: File '{INPUT_FILE}' not found!")
    exit(1)

# Ensure timestamp is datetime
if 'login_timestamp' not in df.columns:
    print("  ERROR: 'login_timestamp' column not found!")
    exit(1)

# Parse timestamp if needed
if df['login_timestamp'].dtype == 'object':
    df['login_timestamp'] = pd.to_datetime(df['login_timestamp'], errors='coerce')
else:
    df['login_timestamp'] = pd.to_datetime(df['login_timestamp'], errors='coerce')

valid_timestamps = df['login_timestamp'].notna().sum()
print(f"  Valid timestamps: {valid_timestamps:,} / {len(df):,}")
if valid_timestamps == 0:
    print("  ERROR: No valid timestamps found!")
    exit(1)
print(f"  Range: {df['login_timestamp'].min()} to {df['login_timestamp'].max()}")

# Sort by timestamp and user_id for deterministic order
df = df.sort_values(['login_timestamp', 'user_id']).reset_index(drop=True)
print(f"  Sorted by timestamp and user_id")

# ============================================================================
# STEP 2: BASIC EXTRACTED FEATURES
# ============================================================================

print("\n[2/7] Creating basic extracted features...")

# Feature 1: time_of_day (hour 0-23)
df['time_of_day'] = df['login_timestamp'].dt.hour
print(f"  Feature 1: time_of_day (range {df['time_of_day'].min()}-{df['time_of_day'].max()})")

# ============================================================================
# STEP 3: IP REPUTATION SCORE
# ============================================================================

print("\n[3/7] Creating IP reputation score...")

# Ensure string types for consistent grouping
df['ip_address'] = df['ip_address'].astype(str)
df['user_id'] = df['user_id'].astype(str)

# Ensure boolean type
if df['is_attack_ip'].dtype != bool:
    df['is_attack_ip'] = df['is_attack_ip'].astype(bool)

# Feature 2: ip_reputation_score = historical attack rate per IP
ip_reputation = df.groupby('ip_address')['is_attack_ip'].mean()
df['ip_reputation_score'] = df['ip_address'].map(ip_reputation).fillna(0.0).astype(float)

print(f"  Feature 2: ip_reputation_score")
print(f"    Range: {df['ip_reputation_score'].min():.3f} to {df['ip_reputation_score'].max():.3f}")
print(f"    IPs with reputation > 0: {(df['ip_reputation_score'] > 0).sum():,}")

# ============================================================================
# STEP 4: GEO ANOMALY FLAG
# ============================================================================

print("\n[4/7] Creating geo anomaly flag...")

# Feature 3: geo_anomaly_flag = 1 if current country != user's most frequent country

def safe_mode(x):
    """Safely get mode of a series. Returns first element if single item, else mode."""
    if len(x) == 0:
        return 'unknown'
    if len(x) == 1:
        return x.iloc[0]
    modes = x.mode()
    if len(modes) > 0:
        return modes.iloc[0]
    return x.iloc[0]

user_typical_country = df.groupby('user_id')['country'].agg(safe_mode)
df['user_typical_country'] = df['user_id'].map(user_typical_country)

# Flag anomaly
df['geo_anomaly_flag'] = (df['country'] != df['user_typical_country']).astype(int)

# Drop helper column
df.drop(columns=['user_typical_country'], inplace=True)

print(f"  Feature 3: geo_anomaly_flag")
print(f"    Anomalies detected: {df['geo_anomaly_flag'].sum():,} ({df['geo_anomaly_flag'].mean()*100:.1f}%)")

# ============================================================================
# STEP 5: ROLLING WINDOW FEATURES
# ============================================================================

print("\n[5/7] Creating rolling window features...")
print("  This may take a moment...")

# Convert timestamp to Unix seconds for efficient comparison
df['ts_numeric'] = df['login_timestamp'].astype('int64') // 10**9

# Device fingerprint for the new-device feature: strip version numbers from
# browser/OS strings (e.g. "Chrome 90.0.4411" -> "Chrome") since raw
# versioned strings are too noisy to match across a user's logins.
def strip_version(value):
    match = re.match(r'^([A-Za-z ]+)', str(value))
    return match.group(1).strip() if match else str(value)

df['browser_name'] = df['browser_name_and_version'].apply(strip_version)
df['os_name'] = df['os_name_and_version'].apply(strip_version)
df['device_fingerprint'] = (
    df['device_type'].astype(str) + '|' + df['browser_name'] + '|' + df['os_name']
)

# Initialize feature columns
df['fail_count_1min'] = 0
df['fail_count_5min'] = 0
df['fail_count_10min'] = 0
df['success_count_1min'] = 0
df['success_count_5min'] = 0
df['unique_usernames_5min'] = 0
df['time_since_last_login'] = np.nan
df['is_new_device'] = 0

# Pre-build user login history for fast lookup
user_last_login = {}

# Per-user set of device fingerprints seen so far
user_devices = defaultdict(set)

# Per-IP history: list of (timestamp, user_id, success)
ip_history = defaultdict(list)

# Convert to list of dicts for fast loop access
rows = df.to_dict('records')

for idx in range(len(rows)):
    row = rows[idx]
    ip = row['ip_address']
    ts = int(row['ts_numeric'])
    user = row['user_id']
    success = bool(row['login_successful'])
    
    # Calculate window boundaries
    window_1min = ts - (WINDOW_1MIN * 60)
    window_5min = ts - (WINDOW_5MIN * 60)
    window_10min = ts - (WINDOW_10MIN * 60)
    
    # Get and prune IP history (keep last 10 minutes)
    history = ip_history[ip]
    history = [h for h in history if h[0] >= window_10min]
    ip_history[ip] = history
    
    # Filter for specific windows
    recent_1min = [h for h in history if h[0] >= window_1min and h[0] < ts]
    recent_5min = [h for h in history if h[0] >= window_5min and h[0] < ts]
    recent_10min = [h for h in history if h[0] >= window_10min and h[0] < ts]
    
    # Count failures and successes
    fails_1min = sum(1 for h in recent_1min if not h[2])
    fails_5min = sum(1 for h in recent_5min if not h[2])
    fails_10min = sum(1 for h in recent_10min if not h[2])
    
    success_1min = sum(1 for h in recent_1min if h[2])
    success_5min = sum(1 for h in recent_5min if h[2])
    
    # Count unique users in 5-min window (INCLUDE current user for real-time accuracy)
    users_in_window = set(h[1] for h in recent_5min)
    users_in_window.add(user)  # FIX: Include current user
    users_5min = len(users_in_window)
    
    # Time since last login for this user (across all IPs)
    if user in user_last_login:
        time_since = ts - user_last_login[user]
    else:
        time_since = NO_PREVIOUS_LOGIN  # FIX: Use sentinel instead of NaN
    
    # Update user last login
    user_last_login[user] = ts

    # New-device check: has this user ever used this device fingerprint before?
    fingerprint = row['device_fingerprint']
    is_new_device = 0 if fingerprint in user_devices[user] else 1
    user_devices[user].add(fingerprint)

    # Assign features
    rows[idx]['fail_count_1min'] = fails_1min
    rows[idx]['fail_count_5min'] = fails_5min
    rows[idx]['fail_count_10min'] = fails_10min
    rows[idx]['success_count_1min'] = success_1min
    rows[idx]['success_count_5min'] = success_5min
    rows[idx]['unique_usernames_5min'] = users_5min
    rows[idx]['time_since_last_login'] = time_since
    rows[idx]['is_new_device'] = is_new_device
    
    # Add current event to IP history
    history.append((ts, user, success))
    
    # Progress
    if (idx + 1) % 2000 == 0:
        print(f"    Processed {idx + 1:,} / {len(rows):,} rows", end='\r', flush=True)

print(f"    Processed {len(rows):,} / {len(rows):,} rows")

# Reconstruct DataFrame
df = pd.DataFrame(rows)

# Drop helper columns
df.drop(columns=['ts_numeric', 'browser_name', 'os_name', 'device_fingerprint'],
        inplace=True, errors='ignore')

print(f"  Feature 4: fail_count_1min (max: {df['fail_count_1min'].max()})")
print(f"  Feature 5: fail_count_5min (max: {df['fail_count_5min'].max()})")
print(f"  Feature 6: unique_usernames_5min (max: {df['unique_usernames_5min'].max()})")
print(f"  Feature 9: time_since_last_login (sentinel: {NO_PREVIOUS_LOGIN})")
print(f"  Feature 11: is_new_device ({df['is_new_device'].sum():,} new-device logins, "
      f"{df['is_new_device'].mean()*100:.1f}%)")

# ============================================================================
# STEP 6: DERIVED RATIO FEATURES
# ============================================================================

print("\n[6/7] Creating derived ratio features...")

# Feature 7: rolling_fail_velocity
df['rolling_fail_velocity'] = df['fail_count_10min'] / 10.0

# Feature 8: login_success_rate_1min
total_1min = df['fail_count_1min'] + df['success_count_1min']
df['login_success_rate_1min'] = np.where(
    total_1min > 0,
    df['success_count_1min'] / total_1min,
    1.0
)

# Feature 10: fail_to_success_ratio_5min
df['fail_to_success_ratio_5min'] = np.where(
    df['success_count_5min'] > 0,
    df['fail_count_5min'] / df['success_count_5min'],
    df['fail_count_5min']  # No successes = ratio = fail count
)
df['fail_to_success_ratio_5min'] = df['fail_to_success_ratio_5min'].replace([np.inf, -np.inf], 999).clip(upper=999)

print(f"  Feature 7: rolling_fail_velocity (max: {df['rolling_fail_velocity'].max():.1f})")
print(f"  Feature 8: login_success_rate_1min (range: {df['login_success_rate_1min'].min():.3f}-{df['login_success_rate_1min'].max():.3f})")
print(f"  Feature 10: fail_to_success_ratio_5min (max: {df['fail_to_success_ratio_5min'].max():.1f})")

# Drop helper columns
df.drop(columns=['success_count_1min', 'success_count_5min', 'fail_count_10min'], inplace=True, errors='ignore')

# ============================================================================
# STEP 7: FINAL CLEANUP AND SAVE
# ============================================================================

print("\n[7/7] Final cleanup and saving...")

# Ensure all feature columns are numeric
feature_cols = [
    'time_of_day',
    'ip_reputation_score',
    'geo_anomaly_flag',
    'fail_count_1min',
    'fail_count_5min',
    'unique_usernames_5min',
    'rolling_fail_velocity',
    'login_success_rate_1min',
    'time_since_last_login',
    'fail_to_success_ratio_5min',
    'is_new_device'
]

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Summary
print(f"\n{'=' * 60}")
print("FEATURE ENGINEERING COMPLETE")
print(f"{'=' * 60}")
print(f"Final shape: {df.shape}")
print(f"Features created: {len(feature_cols)}")

print(f"\nFeature statistics:")
print(df[feature_cols].describe().round(2))

# Save
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to: {OUTPUT_FILE}")

# Verify
try:
    verify = pd.read_csv(OUTPUT_FILE, nrows=5)
    print(f"Verified: {len(verify)} rows readable, {len(verify.columns)} columns")
except Exception as e:
    print(f"WARNING: Could not verify: {e}")

elapsed = time.time() - start_time
print(f"\nTotal time: {elapsed:.1f}s")
print(f"{'=' * 60}")