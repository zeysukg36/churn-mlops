import pandas as pd

# Ham verinin doğru yoldan okunması (data klasörü bir üst dizinde olduğu için ../data/)
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("--- VERİ BOYUTU ---")
print(df.shape)

print("\n--- SÜTUNLAR VE VERİ TİPLERİ ---")
print(df.info())

print("\n--- HEDEF DEĞİŞKEN (CHURN) DAĞILIMI (Sınıf Dengesizliği Kontrolü) ---")
print(df['Churn'].value_counts(normalize=True))
