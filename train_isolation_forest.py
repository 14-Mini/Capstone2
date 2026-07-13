import pandas as pd
import numpy as np
import pickle
import time
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"
TEST_FILE = "test.csv"
MODEL_OUTPUT = "isolation_forest.pkl"

# Features to use (exclude targets and raw labels)
FEATURE_COLS = [
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

# Columns to exclude from features (data leakage targets)
DROP_COLS = ['is_account_takeover', 'is_attack_ip', 'attack_type', 'is_anomaly']

RANDOM_STATE = 42

print("=" * 60)
print("Isolation Forest Training")
print(f"Train: {TRAIN_FILE}")
print(f"Scikit-learn version: {sklearn.__version__}")
print("=" * 60)

start_time = time.time()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[1/7] Loading data...")

try:
    df_train = pd.read_csv(TRAIN_FILE, encoding='utf-8', low_memory=False)
    df_val = pd.read_csv(VAL_FILE, encoding='utf-8', low_memory=False)
    df_test = pd.read_csv(TEST_FILE, encoding='utf-8', low_memory=False)
    print(f"  Train: {len(df_train):,} rows")
    print(f"  Val:   {len(df_val):,} rows")
    print(f"  Test:  {len(df_test):,} rows")
except FileNotFoundError as e:
    print(f"  ERROR: {e}")
    exit(1)

# Verify required columns exist before using them
required_cols = FEATURE_COLS + ['attack_type', 'is_anomaly']
missing_train = [c for c in required_cols if c not in df_train.columns]
missing_val = [c for c in required_cols if c not in df_val.columns]
missing_test = [c for c in required_cols if c not in df_test.columns]

if missing_train:
    print(f"  ERROR: Train missing columns: {missing_train}")
    exit(1)
if missing_val:
    print(f"  ERROR: Val missing columns: {missing_val}")
    exit(1)
if missing_test:
    print(f"  ERROR: Test missing columns: {missing_test}")
    exit(1)

# ============================================================================
# STEP 2: PREPARE TRAINING DATA (NORMAL ONLY)
# ============================================================================

print("\n[2/7] Preparing training data (normal events only)...")

# Verify attack_type exists (already checked above, but defensive)
if 'attack_type' not in df_train.columns:
    print("  ERROR: 'attack_type' column not found in training data!")
    exit(1)

# Filter to normal events only — Isolation Forest is unsupervised
df_train_normal = df_train[df_train['attack_type'] == 'normal']

print(f"  Total train rows: {len(df_train):,}")
print(f"  Normal rows:      {len(df_train_normal):,}")
print(f"  Attack rows:      {len(df_train) - len(df_train_normal):,} (excluded from training)")

if len(df_train_normal) == 0:
    print("  ERROR: No normal rows found in training data!")
    print("  Isolation Forest requires normal data for unsupervised training.")
    exit(1)

if len(df_train_normal) < 100:
    print(f"  WARNING: Only {len(df_train_normal)} normal rows — may be insufficient for training")

# Extract features
X_train = df_train_normal[FEATURE_COLS]

# Verify no missing values
missing = X_train.isnull().sum().sum()
if missing > 0:
    print(f"  Filling {missing} missing values with 0")
    X_train = X_train.fillna(0)

# Verify no infinite values
inf_count = np.isinf(X_train.values).sum()
if inf_count > 0:
    print(f"  WARNING: {inf_count} infinite values found — replacing with 0")
    X_train = X_train.replace([np.inf, -np.inf], 0)

print(f"  Feature matrix shape: {X_train.shape}")

# ============================================================================
# STEP 3: TRAIN INITIAL ISOLATION FOREST
# ============================================================================

print("\n[3/7] Training initial Isolation Forest...")

# Explicit max_samples for deterministic behavior across scikit-learn versions
max_samples = min(256, len(X_train))

model = IsolationForest(
    n_estimators=100,
    max_samples=max_samples,
    contamination=0.13,  # Starting point, will tune
    max_features=1.0,
    bootstrap=False,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0  # Clean output
)

model.fit(X_train)
print(f"  Initial training complete")
print(f"  max_samples used: {max_samples}")

# ============================================================================
# STEP 4: VALIDATE AND TUNE CONTAMINATION
# ============================================================================

print("\n[4/7] Validating and tuning contamination parameter...")

# Prepare validation features
X_val = df_val[FEATURE_COLS].copy()
X_val = X_val.fillna(0).replace([np.inf, -np.inf], 0)

# Ground truth from validation set
y_val = df_val['is_anomaly'].values

# Test multiple contamination values
contamination_values = [0.05, 0.08, 0.10, 0.13, 0.15, 0.20, 0.25]
results = []

best_f1 = -1.0  # Start below possible F1
best_contamination = 0.13
best_model = model

for test_contamination in contamination_values:
    print(f"  Testing contamination={test_contamination:.2f}...", end=' ')
    
    temp_model = IsolationForest(
        n_estimators=100,
        max_samples=max_samples,
        contamination=test_contamination,
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    temp_model.fit(X_train)
    
    # Predict: -1 = anomaly, 1 = normal
    temp_pred = temp_model.predict(X_val)
    temp_binary = np.where(temp_pred == -1, 1, 0)
    
    # Calculate metrics
    temp_accuracy = accuracy_score(y_val, temp_binary)
    temp_precision = precision_score(y_val, temp_binary, zero_division=np.nan)
    temp_recall = recall_score(y_val, temp_binary, zero_division=np.nan)
    temp_f1 = f1_score(y_val, temp_binary, zero_division=np.nan)
    
    results.append({
        'contamination': test_contamination,
        'accuracy': temp_accuracy,
        'precision': temp_precision,
        'recall': temp_recall,
        'f1': temp_f1,
        'predicted_anomalies': temp_binary.sum()
    })
    
    print(f"F1={temp_f1:.4f}, Precision={temp_precision:.4f}, Recall={temp_recall:.4f}, "
          f"Anomalies={temp_binary.sum()}")
    
    # Update best if F1 improved (handle NaN safely)
    if not np.isnan(temp_f1) and temp_f1 > best_f1:
        best_f1 = temp_f1
        best_contamination = test_contamination
        best_model = temp_model

# Print comparison table
print(f"\n  Contamination tuning results:")
print(f"  {'Contam':>8s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'Pred_Anom':>10s}")
for r in results:
    print(f"  {r['contamination']:8.2f} {r['accuracy']:10.4f} "
          f"{r['precision'] if not np.isnan(r['precision']) else 0:10.4f} "
          f"{r['recall'] if not np.isnan(r['recall']) else 0:8.4f} "
          f"{r['f1'] if not np.isnan(r['f1']) else 0:8.4f} "
          f"{r['predicted_anomalies']:10,}")

print(f"\n  Best contamination: {best_contamination:.2f} (F1={best_f1:.4f})")

# Use best model for final evaluation
model = best_model

# Final predictions
val_predictions = model.predict(X_val)
val_binary = np.where(val_predictions == -1, 1, 0)

print(f"\n  Final validation metrics:")
print(f"    Predicted anomalies: {val_binary.sum():,} / {len(val_binary):,}")
print(f"    Actual anomalies:    {y_val.sum():,} / {len(y_val):,}")
print(f"    Accuracy:  {accuracy_score(y_val, val_binary):.4f}")
print(f"    Precision: {precision_score(y_val, val_binary, zero_division=0):.4f}")
print(f"    Recall:    {recall_score(y_val, val_binary, zero_division=0):.4f}")
print(f"    F1-Score:  {f1_score(y_val, val_binary, zero_division=0):.4f}")

print(f"\n  Confusion matrix:")
cm = confusion_matrix(y_val, val_binary)
print(f"                 Predicted")
print(f"                 Normal  Anomaly")
print(f"  Actual Normal  {cm[0,0]:6,}  {cm[0,1]:7,}")
print(f"  Actual Anomaly {cm[1,0]:6,}  {cm[1,1]:7,}")

# Warning if model predicts nothing or everything
if val_binary.sum() == 0:
    print(f"\n  WARNING: Model predicted ZERO anomalies — too conservative!")
elif val_binary.sum() == len(val_binary):
    print(f"\n  WARNING: Model predicted ALL anomalies — too sensitive!")

# ============================================================================
# STEP 5: EVALUATE ON HELD-OUT TEST SET
# ============================================================================

print("\n[5/7] Evaluating on held-out test set...")

X_test = df_test[FEATURE_COLS].copy()
X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)
y_test = df_test['is_anomaly'].values

test_pred = model.predict(X_test)
test_binary = np.where(test_pred == -1, 1, 0)
test_scores = -model.score_samples(X_test)  # higher = more anomalous

test_accuracy = accuracy_score(y_test, test_binary)
test_precision = precision_score(y_test, test_binary, zero_division=0)
test_recall = recall_score(y_test, test_binary, zero_division=0)
test_f1 = f1_score(y_test, test_binary, zero_division=0)
test_roc_auc = roc_auc_score(y_test, test_scores)

print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print(f"  ROC-AUC:   {test_roc_auc:.4f}")

test_cm = confusion_matrix(y_test, test_binary)
print(f"\n  Confusion matrix:")
print(f"                 Predicted")
print(f"                 Normal  Anomaly")
print(f"  Actual Normal  {test_cm[0,0]:6,}  {test_cm[0,1]:7,}")
print(f"  Actual Anomaly {test_cm[1,0]:6,}  {test_cm[1,1]:7,}")

fpr, tpr, _ = roc_curve(y_test, test_scores)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'Isolation Forest (AUC = {test_roc_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Isolation Forest ROC Curve (Test Set)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('isolation_forest_roc_auc.png', dpi=150)
plt.close()
print(f"\n  Saved: isolation_forest_roc_auc.png")

# ============================================================================
# STEP 6: SAVE MODEL
# ============================================================================

print("\n[6/7] Saving model...")

model_data = {
    'model': model,
    'feature_cols': FEATURE_COLS,
    'contamination': best_contamination,
    'n_estimators': 100,
    'max_samples': max_samples,
    'sklearn_version': sklearn.__version__,
    'training_rows': len(X_train),
    'training_features': list(FEATURE_COLS),
    'test_metrics': {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'roc_auc': test_roc_auc
    }
}

with open(MODEL_OUTPUT, 'wb') as f:
    pickle.dump(model_data, f)

print(f"  Saved: {MODEL_OUTPUT}")
print(f"  Metadata: contamination={best_contamination}, n_estimators=100, "
      f"max_samples={max_samples}, sklearn={sklearn.__version__}")

# ============================================================================
# STEP 6: VERIFY SAVED MODEL
# ============================================================================

print("\n[7/7] Verifying saved model...")

try:
    with open(MODEL_OUTPUT, 'rb') as f:
        loaded = pickle.load(f)
    
    # Verify structure
    assert 'model' in loaded, "Missing 'model' key"
    assert 'feature_cols' in loaded, "Missing 'feature_cols' key"
    assert loaded['sklearn_version'] == sklearn.__version__, \
        f"Version mismatch: saved={loaded['sklearn_version']}, current={sklearn.__version__}"
    
    # Test prediction consistency
    loaded_model = loaded['model']
    loaded_pred = loaded_model.predict(X_val.head(10))
    original_pred = model.predict(X_val.head(10))
    
    assert np.array_equal(loaded_pred, original_pred), "Prediction mismatch after loading!"
    
    print(f"  Verification PASSED")
    print(f"    Model type: {type(loaded_model).__name__}")
    print(f"    Features: {len(loaded['feature_cols'])}")
    print(f"    Predictions consistent: Yes")
    
except Exception as e:
    print(f"  Verification FAILED: {e}")
    print(f"  The saved model may be corrupt or incompatible.")

elapsed = time.time() - start_time
print(f"\n{'=' * 60}")
print("COMPLETE")
print(f"Total time: {elapsed:.1f}s")
print(f"{'=' * 60}")