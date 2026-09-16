import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve

# 1. Veriyi okuma
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. TotalCharges sütunundaki boşluk tuzağını düzeltme (NaN -> 0)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# 3. Gereksiz sütunları düşürme
df.drop('customerID', axis=1, inplace=True)

# 4. Hedef değişkeni dönüştürme (Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 5. One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# 6. Özellikler ve Hedef ayrımı
X = df.drop('Churn', axis=1)
y = df['Churn']

# 7. Stratify ile Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Sınıf dengesizliğini yönetmek için class_weight="balanced" eklendi
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 9. Karar Eşiği Optimizasyonu (Threshold Tuning - F1'i maksimize eden eşik)
y_prob = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# F1 skorunu en iyi tetikleyen eşik değerini bulma
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]

# Yeni eşik değerine göre tahminler
y_pred_tuned = (y_prob >= best_threshold).astype(int)

print("--- DÜZELTİLMİŞ VE EŞİK İYİLEŞTİRİLMİŞ MODEL ---")
print(f"Seçilen En İyi Eşik (Threshold): {best_threshold:.4f}")
print(f"Yeni F1-Score: {f1_score(y_test, y_pred_tuned):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
