import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# 1. Veriyi okuma
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. TotalCharges sütunundaki boşluk tuzağını temizleme ve sayısal tipe çevirme
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Eksik kalan satırları median ile doldurma veya düşürme (şimdilik drop ediyoruz)
df.dropna(inplace=True)

# 3. Gereksiz sütunları düşürme (customerID model için anlam taşımaz)
df.drop('customerID', axis=1, inplace=True)

# 4. Hedef değişkeni (Churn) sayısal formata çevirme (Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 5. Kategorik değişkenleri One-Hot Encoding ile dönüştürme
df = pd.get_dummies(df, drop_first=True)

# 6. Özellikler (X) ve Hedef (y) ayrımı
X = df.drop('Churn', axis=1)
y = df['Churn']

# 7. Stratify ile Train/Test Split (Sınıf dengesini koruyarak bölme)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. İlk Baseline Model (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 9. Değerlendirme
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("--- MODEL PERFORMANSI ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

