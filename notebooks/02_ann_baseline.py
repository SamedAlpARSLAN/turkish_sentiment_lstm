import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# =====================
# 1) Dizin ayarları
# =====================
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

ENCODING = "cp1254"  # bir önceki scriptte bulmuştuk

train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"

print("Train path:", train_path)
print("Test path :", test_path)

# =====================
# 2) Veriyi oku
# =====================
train_df = pd.read_csv(train_path, encoding=ENCODING)
test_df = pd.read_csv(test_path, encoding=ENCODING)

# Gereksiz index kolonunu at
for df in (train_df, test_df):
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

print("\nTrain kolonları:", train_df.columns)
print("Test kolonları :", test_df.columns)

# Beklenen kolonlar: comment, Label
assert "comment" in train_df.columns, "comment kolonu bulunamadı"
assert "Label" in train_df.columns, "Label kolonu bulunamadı"

# Eksik satırları temizle
train_df = train_df.dropna(subset=["comment", "Label"])
test_df = test_df.dropna(subset=["comment", "Label"])

print("\nLabel değerleri (train):")
print(train_df["Label"].value_counts())

# =====================
# 3) Metin ön işleme
# =====================

def clean_text(text: str) -> str:
    text = str(text).lower()
    # Türkçe karakterleri de kabul edip noktalama vs'yi siliyoruz
    text = re.sub(r"[^0-9a-zçğıöşü\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_df["clean"] = train_df["comment"].apply(clean_text)
test_df["clean"] = test_df["comment"].apply(clean_text)

print("\nÖrnek temiz metin:")
print(train_df[["comment", "clean"]].head())

# =====================
# 4) Train / Validation ayır
# =====================

X = train_df["clean"].values
y = train_df["Label"].values  # 0 = negatif, 1 = pozitif

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTrain boyutu :", len(X_train))
print("Val boyutu   :", len(X_val))

# =====================
# 5) TF-IDF + ANN (MLPClassifier)
# =====================

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

print("\nTF-IDF matris boyutu (train):", X_train_vec.shape)

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=15,      # çok uzun sürmesin diye düşük tuttuk
    random_state=42,
    verbose=True,
)

print("\n--- ANN (MLP) eğitimi başlıyor ---")
mlp.fit(X_train_vec, y_train)
print("--- Eğitim bitti ---")

# =====================
# 6) Değerlendirme
# =====================

y_val_pred = mlp.predict(X_val_vec)

print("\n--- Validation classification report ---")
print(classification_report(y_val, y_val_pred, digits=4))

val_acc = accuracy_score(y_val, y_val_pred)
print("Validation accuracy:", val_acc)

# Confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negatif (0)", "Pozitif (1)"],
    yticklabels=["Negatif (0)", "Pozitif (1)"],
)
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("ANN (TF-IDF + MLP) Confusion Matrix")
plt.tight_layout()
cm_path = FIG_DIR / "ann_confusion_matrix.png"
plt.savefig(cm_path, dpi=300)
plt.close()
print("Confusion matrix kaydedildi:", cm_path)

# Loss eğrisi (MLPClassifier.loss_curve_)
plt.figure(figsize=(6, 4))
plt.plot(mlp.loss_curve_)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ANN (MLP) Eğitim Loss Eğrisi")
plt.tight_layout()
loss_path = FIG_DIR / "ann_loss_curve.png"
plt.savefig(loss_path, dpi=300)
plt.close()
print("Loss grafiği kaydedildi:", loss_path)

# =====================
# 7) (Opsiyonel) Test seti değerlendirmesi
# =====================

if "Label" in test_df.columns:
    X_test = test_df["clean"].values
    y_test = test_df["Label"].values

    X_test_vec = vectorizer.transform(X_test)
    y_test_pred = mlp.predict(X_test_vec)

    print("\n--- Test classification report ---")
    print(classification_report(y_test, y_test_pred, digits=4))

    test_acc = accuracy_score(y_test, y_test_pred)
    print("Test accuracy:", test_acc)
else:
    print("\nTest setinde Label kolonu yok, test değerlendirmesi atlandı.")
