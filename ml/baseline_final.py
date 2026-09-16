import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve

# 1. Veriyi okuma
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. TotalCharges düzeltmesi ve tip kontrolü (Mentörün şüphesini gidermek için)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
print(f"--- VERİ TİPİ KONTROLÜ ---")
print(f"TotalCharges Veri Tipi: {df['TotalCharges'].dtype}\n") 

# 3. Gereksiz sütunları düşürme
df.drop('customerID', axis=1, inplace=True)

# 4. Hedef değişkeni dönüştürme (Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 5. One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)
# MLOps Kritik Adım: Eğitimde kullanılan sütun sırasını API için sakla
expected_features = df.drop('Churn', axis=1).columns.tolist()

# 6. Özellikler ve Hedef ayrımı
X = df.drop('Churn', axis=1)
y = df['Churn']

# 7. ÜÇLÜ SPLIT (Train / Validation / Test) - Sızıntıyı Önleme
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 8. Model Eğitimi (Sadece Train seti ile)
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 9. Threshold Optimizasyonu (SADECE Validation seti ile)
y_val_prob = model.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_threshold = float(thresholds[np.argmax(f1_scores)]) # JSON serileştirme için float() 

# 10. Gerçek Performans (SADECE Test seti ile - Modelin ilk kez gördüğü veri)
y_test_prob = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= best_threshold).astype(int)

final_f1 = float(f1_score(y_test, y_test_pred))
final_roc_auc = float(roc_auc_score(y_test, y_test_prob))

print("--- DOĞRULANMIŞ MODEL PERFORMANSI (TEST SETİ) ---")
print(f"Validation'dan Seçilen Eşik: {best_threshold:.4f}")
print(f"Gerçek F1-Score: {final_f1:.4f}")
print(f"Gerçek ROC-AUC: {final_roc_auc:.4f}\n")

# 11. MODEL VE METADATA SERİLEŞTİRME (Gün 8 API Hazırlığı)
# Modeli kaydet
joblib.dump(model, 'rf_model.pkl')

# API'nin okuyacağı konfigürasyon dosyasını (metadata) oluştur
metadata = {
    "model_version": "1.0",
    "best_threshold": best_threshold,
    "metrics": {
        "f1_score": final_f1,
        "roc_auc": final_roc_auc
    },
    "expected_features": expected_features
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("✅ rf_model.pkl ve model_metadata.json başarıyla kaydedildi! API katmanına geçişe hazırız.")
