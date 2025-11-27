import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

ENCODING = "cp1254"

train_path = DATA_DIR / "train.csv"
print("Train path:", train_path)

df = pd.read_csv(train_path, encoding=ENCODING)
df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

# 1) Label dağılımı (negatif / pozitif)
label_counts = df["Label"].value_counts().sort_index()
print("\nLabel dağılımı:")
print(label_counts)

plt.figure(figsize=(4, 4))
sns.barplot(x=label_counts.index, y=label_counts.values)
plt.xticks([0, 1], ["Negatif (0)", "Pozitif (1)"])
plt.ylabel("Örnek sayısı")
plt.title("Veri Setinde Sınıf Dağılımı")
plt.tight_layout()
fig1_path = FIG_DIR / "dataset_label_distribution.png"
plt.savefig(fig1_path, dpi=300)
plt.close()
print("Sınıf dağılımı grafiği kaydedildi:", fig1_path)

# 2) Yorum uzunluğu histogramı (kelime sayısı)
df["length"] = df["comment"].astype(str).str.split().str.len()

plt.figure(figsize=(5, 4))
sns.histplot(df["length"], bins=30)
plt.xlabel("Yorum uzunluğu (kelime)")
plt.ylabel("Frekans")
plt.title("Yorum Uzunluklarının Dağılımı")
plt.tight_layout()
fig2_path = FIG_DIR / "dataset_comment_length_hist.png"
plt.savefig(fig2_path, dpi=300)
plt.close()
print("Yorum uzunluğu histogramı kaydedildi:", fig2_path)
