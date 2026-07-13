import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "rba_sample_16k_engineered.csv"
OUTPUT_DIR = "./"

# DATA-DRIVEN THRESHOLDS (based on your distribution analysis)
# Lowered to capture more realistic attack patterns

BRUTE_FORCE_FAIL_1MIN = 3
BRUTE_FORCE_MAX_USERS = 5

CRED_STUFF_UNIQUE_USERS = 6
CRED_STUFF_FAIL_5MIN = 4

DICT_FAIL_5MIN = 3
DICT_FAIL_1MIN_MAX = 2
DICT_MAX_USERS = 8               # separate from BRUTE_FORCE_MAX_USERS: dictionary attacks
                                  # can spread across more accounts than a brute force burst

# Account takeover: since is_account_takeover is rare/empty in sample,
# derive from behavioral pattern instead
ATO_GEO_ANOMALY = True
ATO_TIME_GAP = 86400             # 24 hours since last login
ATO_FAIL_COUNT = 0               # Successful login after anomaly

# Sentinel from feature_engineering.py's NO_PREVIOUS_LOGIN -- must stay in sync.
# A user's very first sampled login has no prior history to compare against,
# so it can't be a "takeover" of an established pattern; excluded explicitly
# below rather than relying on the gap/device checks (999999 > ATO_TIME_GAP,
# so it would otherwise pass the gap check trivially).
NO_PREVIOUS_LOGIN = 999999

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

print("=" * 60)
print("Attack Labeling and Data Split - REVISED THRESHOLDS")
print(f"Input: {INPUT_FILE}")
print("=" * 60)

start_time = time.time()

# ============================================================================
# STEP 1: LOAD ENGINEERED DATA
# ============================================================================

print("\n[1/5] Loading engineered data...")

try:
    df = pd.read_csv(INPUT_FILE, encoding='utf-8', low_memory=False)
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
except FileNotFoundError:
    print(f"  ERROR: File '{INPUT_FILE}' not found!")
    exit(1)

if len(df) == 0:
    print("  ERROR: Loaded DataFrame is empty!")
    exit(1)

required_cols = ['fail_count_1min', 'fail_count_5min', 'unique_usernames_5min',
                 'is_account_takeover', 'login_successful', 'geo_anomaly_flag',
                 'time_since_last_login', 'is_new_device']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print(f"  ERROR: Missing required columns: {missing_cols}")
    exit(1)

# Convert types
if df['is_account_takeover'].dtype != bool:
    df['is_account_takeover'] = df['is_account_takeover'].astype(bool)

ato_count = df['is_account_takeover'].sum()
print(f"  Account takeovers in raw data: {ato_count:,}")

# ============================================================================
# STEP 2: ANALYZE CURRENT DISTRIBUTION (for threshold tuning)
# ============================================================================

print("\n[2/5] Analyzing feature distributions for threshold tuning...")

print(f"  fail_count_1min:     min={df['fail_count_1min'].min()}, max={df['fail_count_1min'].max()}, "
      f"mean={df['fail_count_1min'].mean():.2f}")
print(f"  fail_count_5min:     min={df['fail_count_5min'].min()}, max={df['fail_count_5min'].max()}, "
      f"mean={df['fail_count_5min'].mean():.2f}")
print(f"  unique_usernames_5min: min={df['unique_usernames_5min'].min()}, "
      f"max={df['unique_usernames_5min'].max()}, mean={df['unique_usernames_5min'].mean():.2f}")

# Show percentiles for data-driven threshold selection
for col in ['fail_count_1min', 'fail_count_5min', 'unique_usernames_5min']:
    p95 = df[col].quantile(0.95)
    p99 = df[col].quantile(0.99)
    print(f"  {col}: 95th percentile={p95:.1f}, 99th percentile={p99:.1f}")

# ============================================================================
# STEP 3: CREATE ATTACK TYPE LABELS (REVISED HEURISTICS)
# ============================================================================

print("\n[3/5] Creating attack type labels with revised thresholds...")

def classify_attack(row):
    """
    Classify each login event into attack type.
    Priority: account_takeover > credential_stuffing > brute_force > dictionary_attack > normal
    """
    
    # Account takeover: confirmed flag OR behavioral pattern
    # Behavioral: geo anomaly + long time gap + successful login (suspicious access)
    if pd.notna(row.get('is_account_takeover')) and row['is_account_takeover'] == True:
        return 'account_takeover'
    
    # Also flag as account takeover if: geo anomaly AND very long time gap
    # This catches takeovers even without confirmed flag
    if (row['geo_anomaly_flag'] == 1 and
        row['time_since_last_login'] > ATO_TIME_GAP and
        row['time_since_last_login'] != NO_PREVIOUS_LOGIN and
        row['login_successful'] == True and
        row['is_new_device'] == 1):
        return 'account_takeover'
    
    # Credential stuffing: multiple users from same IP, some failures
    # Lowered threshold: 2+ unique users (was 5)
    if (row['unique_usernames_5min'] >= CRED_STUFF_UNIQUE_USERS and 
        row['fail_count_5min'] >= CRED_STUFF_FAIL_5MIN):
        return 'credential_stuffing'
    
    # Brute force: repeated failures on same user(s)
    # Lowered threshold: 3+ failures in 1 minute (was 10)
    if (row['fail_count_1min'] >= BRUTE_FORCE_FAIL_1MIN and 
        row['unique_usernames_5min'] <= BRUTE_FORCE_MAX_USERS):
        return 'brute_force'
    
    # Dictionary attack: many failures over time, slower pace
    if (row['fail_count_5min'] >= DICT_FAIL_5MIN and
        row['fail_count_1min'] <= DICT_FAIL_1MIN_MAX and
        row['unique_usernames_5min'] <= DICT_MAX_USERS):
        return 'dictionary_attack'
    
    return 'normal'

df['attack_type'] = df.apply(classify_attack, axis=1)

# Show distribution
attack_counts = df['attack_type'].value_counts()
print(f"\n  Attack type distribution:")
for attack, count in attack_counts.items():
    pct = count / len(df) * 100
    print(f"    {attack:20s}: {count:5,} ({pct:5.1f}%)")

# ============================================================================
# STEP 4: CREATE BINARY ANOMALY LABEL
# ============================================================================

print("\n[4/5] Creating binary anomaly label...")

df['is_anomaly'] = (df['attack_type'] != 'normal').astype(int)

anomaly_count = df['is_anomaly'].sum()
normal_count = (df['is_anomaly'] == 0).sum()
print(f"  Normal:     {normal_count:,} ({normal_count/len(df)*100:.1f}%)")
print(f"  Anomaly:    {anomaly_count:,} ({anomaly_count/len(df)*100:.1f}%)")

# Warn if still imbalanced
anomaly_rate = anomaly_count / len(df)
if anomaly_rate > 0.5:
    print(f"  WARNING: Anomaly rate is {anomaly_rate:.1%} — very high!")
elif anomaly_rate < 0.05:
    print(f"  WARNING: Anomaly rate is {anomaly_rate:.1%} — still very low!")
    print(f"  Consider lowering thresholds further if models underperform.")
else:
    print(f"  Anomaly rate {anomaly_rate:.1%} is acceptable range (5-50%)")

# ============================================================================
# STEP 5: TRAIN / VALIDATION / TEST SPLIT
# ============================================================================

print("\n[5/5] Splitting into train / validation / test sets...")

can_stratify = attack_counts.min() >= 2
stratify_col = df['attack_type'] if can_stratify else None

if not can_stratify:
    print(f"  Stratification disabled (smallest class has {attack_counts.min()} samples)")

# First split: test set (15%)
try:
    df_temp, df_test = train_test_split(
        df, 
        test_size=TEST_RATIO, 
        random_state=RANDOM_STATE, 
        stratify=stratify_col
    )
    print(f"  Test split: OK ({len(df_test):,} rows)")
except ValueError as e:
    print(f"  Test split failed: {e}")
    print(f"  Using random split...")
    df_temp, df_test = train_test_split(df, test_size=TEST_RATIO, random_state=RANDOM_STATE)

# Second split: validation from remaining
val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
temp_counts = df_temp['attack_type'].value_counts()
can_stratify_temp = temp_counts.min() >= 2
stratify_temp = df_temp['attack_type'] if can_stratify_temp else None

try:
    df_train, df_val = train_test_split(
        df_temp,
        test_size=val_ratio_adjusted,
        random_state=RANDOM_STATE,
        stratify=stratify_temp
    )
    print(f"  Validation split: OK ({len(df_val):,} rows)")
except ValueError as e:
    print(f"  Validation split failed: {e}")
    print(f"  Using random split...")
    df_train, df_val = train_test_split(df_temp, test_size=val_ratio_adjusted, random_state=RANDOM_STATE)

# Verify
total_split = len(df_train) + len(df_val) + len(df_test)
if total_split != len(df):
    print(f"\n  WARNING: Split sizes don't match! Diff: {len(df) - total_split}")
else:
    print(f"  Split sizes verified: {total_split:,}")

print(f"\n  Final split:")
print(f"    Train:      {len(df_train):,} rows ({len(df_train)/len(df)*100:.1f}%)")
print(f"    Validation: {len(df_val):,} rows ({len(df_val)/len(df)*100:.1f}%)")
print(f"    Test:       {len(df_test):,} rows ({len(df_test)/len(df)*100:.1f}%)")

# Verify stratification
print(f"\n  Stratification check (attack_type %):")
for split_name, split_df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
    dist = split_df['attack_type'].value_counts(normalize=True) * 100
    print(f"    {split_name:12s}: normal={dist.get('normal', 0):.1f}%, "
          f"brute_force={dist.get('brute_force', 0):.1f}%, "
          f"cred_stuff={dist.get('credential_stuffing', 0):.1f}%, "
          f"dict={dist.get('dictionary_attack', 0):.1f}%, "
          f"ato={dist.get('account_takeover', 0):.1f}%")

# ============================================================================
# SAVE
# ============================================================================

print(f"\n{'=' * 60}")
print("SAVING FILES")
print(f"{'=' * 60}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df_train.to_csv(f"{OUTPUT_DIR}train.csv", index=False)
df_val.to_csv(f"{OUTPUT_DIR}val.csv", index=False)
df_test.to_csv(f"{OUTPUT_DIR}test.csv", index=False)
df.to_csv(f"{OUTPUT_DIR}full_labeled.csv", index=False)

print(f"  Saved: train.csv, val.csv, test.csv, full_labeled.csv")

print(f"\n  IMPORTANT: Exclude these columns from model training:")
print(f"    - is_account_takeover, is_attack_ip (ground truth)")
print(f"    - attack_type, is_anomaly (target labels)")

elapsed = time.time() - start_time
print(f"\n{'=' * 60}")
print("COMPLETE")
print(f"Total time: {elapsed:.1f}s")
print(f"{'=' * 60}")