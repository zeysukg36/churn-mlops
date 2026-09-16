import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df = df.drop(columns=["customerID"])

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough",
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, class_weight="balanced")),
])

pipeline.fit(X_train, y_train)

# --- DÜZELTME: Off-by-one (sentinel nokta) hizalama düzeltmesi ---
y_val_prob = pipeline.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
best_threshold = float(thresholds[np.argmax(f1_scores)])

y_test_prob = pipeline.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= best_threshold).astype(int)
final_f1 = f1_score(y_test, y_test_pred)
final_auc = roc_auc_score(y_test, y_test_prob)

print("--- HİZALANMIŞ (DÜZELTİLMİŞ) PIPELINE SONUÇLARI ---")
print(f"Seçilen Eşik (Validation'dan): {best_threshold:.4f}")
print(f"Gerçek F1-Score (Test): {final_f1:.4f}")
print(f"Gerçek ROC-AUC (Test): {final_auc:.4f}")

joblib.dump(pipeline, "rf_model.pkl")

metadata = {
    "version": "1.1.1",
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "best_threshold": round(best_threshold, 4),
    "metrics": {
        "f1_score": round(final_f1, 4),
        "roc_auc": round(final_auc, 4),
    },
    "categorical_columns": categorical_cols,
    "numeric_columns": numeric_cols,
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Düzeltilmiş pipeline ve metadata başarıyla güncellendi.")
