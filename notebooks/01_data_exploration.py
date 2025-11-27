import pandas as pd
from pathlib import Path

# --- Dizin ayarları ---
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

train_path = DATA_DIR / "train.csv"
test_path  = DATA_DIR / "test.csv"

print("Train path :", train_path)
print("Test path  :", test_path)

if not train_path.exists():
    raise FileNotFoundError(f"train.csv bulunamadı: {train_path}")

if not test_path.exists():
    raise FileNotFoundError(f"test.csv bulunamadı: {test_path}")

# --- Farklı encoding dene ---
encodings = ["utf-8", "cp1254", "iso-8859-9", "latin1"]

train_df = None
test_df = None
used_enc = None

for enc in encodings:
    try:
        print(f"\nEncoding deneniyor: {enc}")
        train_df = pd.read_csv(train_path, encoding=enc)
        test_df  = pd.read_csv(test_path, encoding=enc)
        used_enc = enc
        print(f"--> Başarılı encoding: {enc}")
        break
    except UnicodeDecodeError as e:
        print(f"Encoding başarısız: {enc} ({e})")

if train_df is None:
    raise RuntimeError("Hiçbir encoding ile CSV okunamadı, dosyanın bozuk olup olmadığını kontrol et.")

# --- Temel bilgiler ---
print("\nKullanılan encoding:", used_enc)

print("\n--- Train ilk 5 satır ---")
print(train_df.head())

print("\n--- Train kolon isimleri ---")
print(train_df.columns)

print("\nTrain shape:", train_df.shape)
print("Test shape :", test_df.shape)

print("\n--- Örnek satırlar ---")
for i in range(3):
    print(f"\nSatır {i}:")
    print(train_df.iloc[i])
