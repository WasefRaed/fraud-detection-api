import pandas as pd
import numpy as np
import joblib
import pickle
import os
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, roc_auc_score
from collections import Counter

# ── 1. Load Data ──────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('creditcard.csv')

# ── 2. Scale Amount and Time ──────────────────────────────────
scaler_amount = RobustScaler()
scaler_time   = RobustScaler()
df['Amount'] = scaler_amount.fit_transform(df[['Amount']])
df['Time']   = scaler_time.fit_transform(df[['Time']])

# ── 3. Feature Engineering ────────────────────────────────────
df['Hour']             = (df['Time'] / 3600) % 24
df['Amount_log']       = np.log1p(df['Amount'].clip(lower=0))
df['V1_V2']            = df['V1'] * df['V2']
df['V1_V3']            = df['V1'] * df['V3']
df['Amount_deviation'] = df['Amount'] - df['Amount'].mean()
df['V1_squared']       = df['V1'] ** 2

amount_bins = pd.cut(df['Amount'], bins=[-np.inf, 50, 100, 200, np.inf], labels=[0,1,2,3])
df['Amount_Category'] = amount_bins.astype(float)

# ── 4. Feature Selection ──────────────────────────────────────
X = df.drop('Class', axis=1)
y = df['Class']

print("Selecting features...")
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
selected_features = mi_df[mi_df > 0.001].index.tolist()
print(f"  Selected {len(selected_features)} features")

X_selected = X[selected_features]

# ── 5. Train/Test Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.3, random_state=42, stratify=y
)

# ── 6. SMOTE ──────────────────────────────────────────────────
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# ── 7. Train XGBoost ──────────────────────────────────────────
print("Training model...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
}

xgb = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
search = RandomizedSearchCV(xgb, param_grid, n_iter=8, cv=3,
                            scoring='f1', random_state=42, verbose=1)
search.fit(X_train_bal, y_train_bal)
best_model = search.best_estimator_

# ── 8. Find Optimal Threshold ─────────────────────────────────
y_proba = best_model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.05)
best_thresh = max(thresholds, key=lambda t: f1_score(y_test, (y_proba >= t).astype(int)))
print(f"  Optimal threshold: {best_thresh:.2f}")
print(f"  ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ── 9. Save Everything ────────────────────────────────────────
os.makedirs('models', exist_ok=True)

joblib.dump(best_model, 'models/model.pkl')
joblib.dump(scaler_amount, 'models/scaler_amount.pkl')
joblib.dump(scaler_time,   'models/scaler_time.pkl')
best_model.get_booster().set_param({'base_score': 0.5})
explainer = shap.TreeExplainer(best_model)
joblib.dump(explainer, 'models/explainer.pkl')
print("✓ SHAP explainer saved")

with open('models/features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

with open('models/params.pkl', 'wb') as f:
    pickle.dump({'threshold': float(best_thresh)}, f)

print("✓ All files saved to models/")