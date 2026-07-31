import pandas as pd
import numpy as np
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "data/interim/rba_sample_16k.csv"      # Output from the sampler
OUTPUT_FILE = "data/interim/rba_sample_16k_cleaned.csv"  # Cleaned output

# ============================================================================
# MAIN SCRIPT
# ============================================================================

print("=" * 60)
print("Data Cleaning - RBA Sample (16K rows)")
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print("=" * 60)

start_time = time.time()

# ============================================================================
# STEP 1: LOAD THE SAMPLED DATA
# ============================================================================

print("\n[1/6] Loading sampled data...")

try:
    df = pd.read_csv(INPUT_FILE, encoding='utf-8', low_memory=False)
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"  ERROR: File '{INPUT_FILE}' not found!")
    print(f"  Make sure the sampler script ran successfully first.")
    exit(1)
except Exception as e:
    print(f"  ERROR loading file: {e}")
    exit(1)

# ============================================================================
# STEP 2: STANDARDIZE COLUMN NAMES
# ============================================================================

print("\n[2/6] Standardizing column names...")

# Store original names for reference
original_columns = list(df.columns)

# Clean column names: lowercase, replace spaces with underscores, remove parentheses
df.columns = (df.columns
              .str.strip()           # Remove leading/trailing whitespace
              .str.lower()           # Convert to lowercase
              .str.replace(' ', '_', regex=False)   # Spaces → underscores
              .str.replace('(', '', regex=False)     # Remove opening parentheses
              .str.replace(')', '', regex=False))    # Remove closing parentheses

print(f"  Original: {original_columns}")
print(f"  Cleaned:  {list(df.columns)}")

# ============================================================================
# STEP 3: HANDLE MISSING VALUES
# ============================================================================

print("\n[3/6] Checking and handling missing values...")

missing = df.isnull().sum()
missing_cols = missing[missing > 0]

if len(missing_cols) > 0:
    print(f"  Found missing values in {len(missing_cols)} columns:")
    
    for col, count in missing_cols.items():
        pct = count / len(df) * 100
        print(f"    {col}: {count:,} rows ({pct:.1f}%)")
    
    # Handle device_type: known issue where unparsed user agents result in empty values
    if 'device_type' in df.columns:
        missing_device = df['device_type'].isnull().sum()
        if missing_device > 0:
            print(f"    Filling {missing_device:,} missing device_type with 'unknown'")
            df['device_type'] = df['device_type'].fillna('unknown')
    
    # Fill remaining numeric columns with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"    Filling {null_count:,} missing values in '{col}' with 0")
            df[col] = df[col].fillna(0)
    
    # Fill remaining text/object columns with 'unknown'
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"    Filling {null_count:,} missing values in '{col}' with 'unknown'")
            df[col] = df[col].fillna('unknown')
    
    # Verify no missing values remain
    remaining_missing = df.isnull().sum().sum()
    print(f"  Remaining missing values after cleanup: {remaining_missing}")
else:
    print("  No missing values found.")

# ============================================================================
# STEP 4: FIX DATA TYPES
# ============================================================================

print("\n[4/6] Fixing data types...")

# --- Timestamp handling ---
# The RBA dataset uses Unix timestamps (seconds since epoch)
# We try unit='s' first; if that gives dates in 1970, we try without unit
if 'login_timestamp' in df.columns:
    print("  Converting 'login_timestamp' to datetime...")
    
    # Attempt 1: Unix seconds
    df['login_timestamp'] = pd.to_datetime(df['login_timestamp'], unit='s', errors='coerce')
    
    # Check if conversion failed (all NaT or dates in 1970)
    valid_timestamps = df['login_timestamp'].notna().sum()
    if valid_timestamps > 0:
        max_year = df['login_timestamp'].dt.year.max()
        if max_year < 2000:
            print(f"    unit='s' gave year {max_year}, trying without unit...")
            df['login_timestamp'] = pd.to_datetime(df['login_timestamp'], errors='coerce')
    
    # Final check
    valid_count = df['login_timestamp'].notna().sum()
    print(f"    Valid timestamps: {valid_count:,} / {len(df):,}")
    if valid_count > 0:
        print(f"    Range: {df['login_timestamp'].min()} to {df['login_timestamp'].max()}")
else:
    print("  WARNING: 'login_timestamp' column not found!")

# --- Boolean columns ---
# These should be True/False but may be loaded as strings or integers
bool_cols = ['login_successful', 'is_attack_ip', 'is_account_takeover']

for col in bool_cols:
    if col in df.columns:
        # Handle various possible representations
        if df[col].dtype == 'object':
            # String values like "True", "False", "true", "false"
            df[col] = df[col].str.strip().str.lower()
            df[col] = df[col].map({'true': True, 'false': False, '1': True, '0': False})
            # Any unmapped values become NaN → fill with False
            df[col] = df[col].fillna(False)
        elif df[col].dtype in ['int64', 'float64']:
            # Numeric 1/0
            df[col] = df[col].astype(bool)
        else:
            # Already boolean or other
            df[col] = df[col].astype(bool)
        
        print(f"  '{col}' → boolean ({df[col].sum():,} True, {(~df[col]).sum():,} False)")
    else:
        print(f"  WARNING: '{col}' column not found!")

# --- ASN (Autonomous System Number) ---
if 'asn' in df.columns:
    print("  Converting 'asn' to integer...")
    # Coerce errors to NaN, fill with 0, convert to int
    df['asn'] = pd.to_numeric(df['asn'], errors='coerce')
    df['asn'] = df['asn'].fillna(0).astype(int)
    print(f"    Range: {df['asn'].min()} to {df['asn'].max()}")
else:
    print("  WARNING: 'asn' column not found!")

# --- RTT (Round-Trip Time) ---
# The original column name might be long; check both possibilities
rtt_col = None
for possible_name in ['round-trip_time_rtt_ms', 'rtt', 'round_trip_time']:
    if possible_name in df.columns:
        rtt_col = possible_name
        break

if rtt_col:
    print(f"  Converting '{rtt_col}' to integer (renaming to 'rtt')...")
    df['rtt'] = pd.to_numeric(df[rtt_col], errors='coerce')
    df['rtt'] = df['rtt'].fillna(0).astype(int)
    
    # Drop the original long-named column if it's different from 'rtt'
    if rtt_col != 'rtt':
        df.drop(columns=[rtt_col], inplace=True)
        print(f"    Dropped original column '{rtt_col}'")
    
    print(f"    Range: {df['rtt'].min()}ms to {df['rtt'].max()}ms")
else:
    print("  WARNING: RTT column not found!")

# --- User ID ---
if 'user_id' in df.columns:
    print("  Converting 'user_id' to string...")
    df['user_id'] = df['user_id'].astype(str)
    print(f"    Unique users: {df['user_id'].nunique():,}")
else:
    print("  WARNING: 'user_id' column not found!")

# --- IP Address ---
if 'ip_address' in df.columns:
    print("  Converting 'ip_address' to string...")
    df['ip_address'] = df['ip_address'].astype(str)
    print(f"    Unique IPs: {df['ip_address'].nunique():,}")
else:
    print("  WARNING: 'ip_address' column not found!")

# ============================================================================
# STEP 5: REMOVE DUPLICATE ROWS
# ============================================================================

print("\n[5/6] Removing duplicate rows...")

before_dedup = len(df)
df = df.drop_duplicates()
after_dedup = len(df)

removed = before_dedup - after_dedup
if removed > 0:
    print(f"  Removed {removed:,} duplicate rows ({removed/before_dedup*100:.2f}%)")
else:
    print("  No duplicate rows found.")

print(f"  Remaining: {after_dedup:,} rows")

# ============================================================================
# STEP 6: CHECK AND FIX ANOMALOUS VALUES
# ============================================================================

print("\n[6/6] Checking for anomalous values...")

# --- RTT anomalies ---
if 'rtt' in df.columns:
    negative_rtt = (df['rtt'] < 0).sum()
    extreme_rtt = (df['rtt'] > 10000).sum()
    
    if negative_rtt > 0:
        print(f"  Found {negative_rtt:,} negative RTT values → setting to 0")
        df.loc[df['rtt'] < 0, 'rtt'] = 0
    
    if extreme_rtt > 0:
        print(f"  Found {extreme_rtt:,} extreme RTT values (>10,000ms) → capping at 10,000")
        df.loc[df['rtt'] > 10000, 'rtt'] = 10000
    
    print(f"  RTT final range: {df['rtt'].min()}ms to {df['rtt'].max()}ms")

# --- Timestamp anomalies ---
if 'login_timestamp' in df.columns:
    future_count = (df['login_timestamp'] > pd.Timestamp.now()).sum()
    if future_count > 0:
        print(f"  Found {future_count:,} future timestamps")
        print(f"    (Expected for synthetic data, no action taken)")

# --- Check for empty strings in key text columns ---
text_cols = ['country', 'region', 'city', 'os', 'browser', 'device_type']
for col in text_cols:
    if col in df.columns:
        empty_strings = (df[col] == '').sum()
        if empty_strings > 0:
            print(f"  Found {empty_strings:,} empty strings in '{col}' → filling with 'unknown'")
            df.loc[df[col] == '', col] = 'unknown'

# ============================================================================
# FINAL SUMMARY AND SAVE
# ============================================================================

print(f"\n{'=' * 60}")
print("CLEANING COMPLETE")
print(f"{'=' * 60}")

print(f"\nFinal dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print(f"\nColumn details:")
for col in df.columns:
    dtype = df[col].dtype
    unique = df[col].nunique()
    print(f"  {col:25s} | {str(dtype):15s} | {unique:6,} unique values")

print(f"\nBoolean column summaries:")
for col in ['login_successful', 'is_attack_ip', 'is_account_takeover']:
    if col in df.columns:
        true_count = df[col].sum()
        false_count = (~df[col]).sum()
        print(f"  {col}: {true_count:,} True, {false_count:,} False")

# Save cleaned data
print(f"\nSaving to: {OUTPUT_FILE}")
df.to_csv(OUTPUT_FILE, index=False)

# Verify save
try:
    verify = pd.read_csv(OUTPUT_FILE, nrows=5)
    print(f"  Verified: {len(verify)} rows readable")
except Exception as e:
    print(f"  WARNING: Could not verify saved file: {e}")

elapsed = time.time() - start_time
print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"{'=' * 60}")