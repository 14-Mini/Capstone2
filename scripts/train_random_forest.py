import pandas as pd
import numpy as np
import pickle
import time
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_FILE = "data/splits/train.csv"
VAL_FILE = "data/splits/val.csv"
TEST_FILE = "data/splits/test.csv"
MODEL_OUTPUT = "models/random_forest.pkl"

TARGET_COL = 'attack_type'

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

# Columns to exclude from features (data leakage / other targets)
DROP_COLS = ['is_account_takeover', 'is_attack_ip', 'attack_type', 'is_anomaly']

RANDOM_STATE = 42

print("=" * 60)
print("Random Forest Training (multiclass attack_type)")
print(f"Train: {TRAIN_FILE}")
print(f"Scikit-learn version: {sklearn.__version__}")
print("=" * 60)

start_time = time.time()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[1/6] Loading data...")

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

required_cols = FEATURE_COLS + [TARGET_COL]
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

print(f"\n  Train class distribution:")
train_dist = df_train[TARGET_COL].value_counts()
for cls, count in train_dist.items():
    print(f"    {cls:20s}: {count:5,} ({count/len(df_train)*100:5.1f}%)")

# ============================================================================
# STEP 2: PREPARE FEATURES / LABELS
# ============================================================================

print("\n[2/6] Preparing features and labels...")

X_train = df_train[FEATURE_COLS].copy()
y_train = df_train[TARGET_COL]

X_val = df_val[FEATURE_COLS].copy()
y_val = df_val[TARGET_COL]

for X in (X_train, X_val):
    missing = X.isnull().sum().sum()
    if missing > 0:
        X.fillna(0, inplace=True)
    inf_count = np.isinf(X.values).sum()
    if inf_count > 0:
        X.replace([np.inf, -np.inf], 0, inplace=True)

print(f"  Feature matrix shapes: train {X_train.shape}, val {X_val.shape}")

# ============================================================================
# STEP 2b: SMOTE OVERSAMPLING (TRAINING DATA ONLY)
# ============================================================================

print("\n[2b/6] Applying SMOTE oversampling to training data...")

smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"  Before SMOTE: {dict(y_train.value_counts())}")
print(f"  After SMOTE:  {dict(pd.Series(y_train_res).value_counts())}")

# ============================================================================
# STEP 3: TRAIN RANDOM FOREST
# ============================================================================

print("\n[3/6] Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    class_weight='balanced',
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# CV uses a SMOTE+RF pipeline so oversampling happens inside each fold's
# training split only — resampling before the split would leak synthetic
# neighbors of validation-fold rows into the training-fold data.
cv_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('rf', RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight='balanced',
        n_jobs=-1,
        random_state=RANDOM_STATE
    ))
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_f1_macro = cross_val_score(cv_pipeline, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
print(f"  5-fold CV F1(macro) [SMOTE + balanced]: {cv_f1_macro.mean():.4f} (+/- {cv_f1_macro.std():.4f})")

model.fit(X_train_res, y_train_res)
print(f"  Training complete ({model.n_estimators} trees, {len(X_train_res):,} rows after SMOTE)")

# ============================================================================
# STEP 4: EVALUATE ON VALIDATION SET
# ============================================================================

print("\n[4/6] Evaluating on validation set...")

val_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, val_pred)
precision_macro = precision_score(y_val, val_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_val, val_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_val, val_pred, average='macro', zero_division=0)
f1_weighted = f1_score(y_val, val_pred, average='weighted', zero_division=0)

print(f"  Accuracy:        {accuracy:.4f}")
print(f"  Precision(macro): {precision_macro:.4f}")
print(f"  Recall(macro):    {recall_macro:.4f}")
print(f"  F1(macro):        {f1_macro:.4f}")
print(f"  F1(weighted):     {f1_weighted:.4f}")

print(f"\n  Classification report:")
print(classification_report(y_val, val_pred, zero_division=0))

labels = sorted(y_val.unique())
cm = confusion_matrix(y_val, val_pred, labels=labels)
print(f"  Confusion matrix (rows=actual, cols=predicted):")
print(f"  Labels: {labels}")
print(cm)

# Feature importance
print(f"\n  Feature importances:")
importances = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
for feat, imp in importances:
    print(f"    {feat:25s}: {imp:.4f}")

# ============================================================================
# STEP 5: EVALUATE ON HELD-OUT TEST SET
# ============================================================================

print("\n[5/6] Evaluating on held-out test set...")

X_test = df_test[FEATURE_COLS].copy()
y_test = df_test[TARGET_COL]
X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)

test_pred = model.predict(X_test)
test_proba = model.predict_proba(X_test)

test_accuracy = accuracy_score(y_test, test_pred)
test_precision_macro = precision_score(y_test, test_pred, average='macro', zero_division=0)
test_recall_macro = recall_score(y_test, test_pred, average='macro', zero_division=0)
test_f1_macro = f1_score(y_test, test_pred, average='macro', zero_division=0)

print(f"  Accuracy:         {test_accuracy:.4f}")
print(f"  Precision(macro): {test_precision_macro:.4f}")
print(f"  Recall(macro):    {test_recall_macro:.4f}")
print(f"  F1(macro):        {test_f1_macro:.4f}")

print(f"\n  Classification report:")
print(classification_report(y_test, test_pred, zero_division=0))

test_labels = sorted(y_test.unique())
test_cm = confusion_matrix(y_test, test_pred, labels=test_labels)
print(f"  Confusion matrix (rows=actual, cols=predicted):")
print(f"  Labels: {test_labels}")
print(test_cm)

# Per-class one-vs-rest ROC-AUC + ROC curves
print(f"\n  Per-class ROC-AUC:")
roc_auc_per_class = {}
plt.figure(figsize=(7, 7))
for i, cls in enumerate(model.classes_):
    y_true_bin = (y_test == cls).astype(int)
    auc = roc_auc_score(y_true_bin, test_proba[:, i])
    roc_auc_per_class[cls] = auc
    print(f"    {cls:20s}: {auc:.4f}")
    fpr, tpr, _ = roc_curve(y_true_bin, test_proba[:, i])
    plt.plot(fpr, tpr, label=f'{cls} (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curves by Attack Class (Test Set)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('reports/random_forest_roc_auc.png', dpi=150)
plt.close()
print(f"\n  Saved: random_forest_roc_auc.png")

# ============================================================================
# STEP 6: SAVE MODEL
# ============================================================================

print("\n[6/6] Saving model...")

model_data = {
    'model': model,
    'feature_cols': FEATURE_COLS,
    'target_col': TARGET_COL,
    'classes': list(model.classes_),
    'n_estimators': model.n_estimators,
    'sklearn_version': sklearn.__version__,
    'training_rows': len(X_train_res),
    'test_metrics': {
        'accuracy': test_accuracy,
        'precision_macro': test_precision_macro,
        'recall_macro': test_recall_macro,
        'f1_macro': test_f1_macro,
        'roc_auc_per_class': roc_auc_per_class
    }
}

with open(MODEL_OUTPUT, 'wb') as f:
    pickle.dump(model_data, f)

print(f"  Saved: {MODEL_OUTPUT}")

# Verify saved model
try:
    with open(MODEL_OUTPUT, 'rb') as f:
        loaded = pickle.load(f)

    assert 'model' in loaded, "Missing 'model' key"
    loaded_model = loaded['model']
    loaded_pred = loaded_model.predict(X_val.head(10))
    original_pred = model.predict(X_val.head(10))
    assert np.array_equal(loaded_pred, original_pred), "Prediction mismatch after loading!"

    print(f"  Verification PASSED")
    print(f"    Model type: {type(loaded_model).__name__}")
    print(f"    Classes: {loaded['classes']}")
except Exception as e:
    print(f"  Verification FAILED: {e}")

elapsed = time.time() - start_time
print(f"\n{'=' * 60}")
print("COMPLETE")
print(f"Total time: {elapsed:.1f}s")
print(f"{'=' * 60}")
