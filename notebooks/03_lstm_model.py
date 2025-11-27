import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

# =====================
# 1) Temel ayarlar
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cihaz:", DEVICE)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

ENCODING = "cp1254"

train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"

print("Train path:", train_path)
print("Test path :", test_path)

# =====================
# 2) Veriyi oku
# =====================
train_df = pd.read_csv(train_path, encoding=ENCODING)
test_df = pd.read_csv(test_path, encoding=ENCODING)

# Gereksiz kolonları at
for df in (train_df, test_df):
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

print("\nTrain kolonları:", train_df.columns)
print("Test kolonları :", test_df.columns)

assert "comment" in train_df.columns
assert "Label" in train_df.columns

train_df = train_df.dropna(subset=["comment", "Label"])
test_df = test_df.dropna(subset=["comment", "Label"])

print("\nLabel değerleri (train):")
print(train_df["Label"].value_counts())

# =====================
# 3) Metin temizleme
# =====================
def clean_text(text: str) -> str:
    text = str(text).lower()
    # Türkçe karakterlere izin ver, noktalama vs temizle
    text = re.sub(r"[^0-9a-zçğıöşü\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_df["clean"] = train_df["comment"].apply(clean_text)
test_df["clean"] = test_df["comment"].apply(clean_text)

print("\nÖrnek temiz metin:")
print(train_df[["comment", "clean"]].head())

# =====================
# 4) Train / Val ayır
# =====================
X_all = train_df["clean"].values
y_all = train_df["Label"].values.astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all,
)

print("\nTrain boyutu :", len(X_train))
print("Val boyutu   :", len(X_val))

# Test seti
X_test = test_df["clean"].values
y_test = test_df["Label"].values.astype(int)

# =====================
# 5) Vocabulary (sözlük) oluşturma
# =====================
def build_vocab(texts, min_freq=2, max_size=20000):
    counter = Counter()
    for text in texts:
        tokens = text.split()
        counter.update(tokens)

    # Özel tokenlar
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)

    return vocab

vocab = build_vocab(X_train, min_freq=2, max_size=20000)
vocab_size = len(vocab)
print("\nVocabulary boyutu:", vocab_size)


def encode_text(text: str, vocab: dict, max_len: int = 100):
    tokens = text.split()
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


MAX_LEN = 100

# =====================
# 6) PyTorch Dataset
# =====================
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])
        ids = encode_text(text, self.vocab, self.max_len)
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


train_dataset = TextDataset(X_train, y_train, vocab, max_len=MAX_LEN)
val_dataset = TextDataset(X_val, y_val, vocab, max_len=MAX_LEN)
test_dataset = TextDataset(X_test, y_test, vocab, max_len=MAX_LEN)

BATCH_SIZE = 64

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =====================
# 7) LSTM Modeli
# =====================
class LSTMSentiment(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, (h_n, c_n) = self.lstm(emb)
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        if self.bidirectional:
            # Son katmanın ileri ve geri yön gizli hallerini birleştir
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h = torch.cat((h_forward, h_backward), dim=1)
        else:
            h = h_n[-1, :, :]
        h = self.dropout(h)
        logits = self.fc(h)  # (batch, num_classes)
        return logits


model = LSTMSentiment(
    vocab_size=vocab_size,
    embed_dim=128,
    hidden_dim=128,
    num_layers=1,
    num_classes=2,
    bidirectional=True,
    dropout=0.3,
).to(DEVICE)

print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# =====================
# 8) Eğitim döngüsü
# =====================
EPOCHS = 5  # CPU için makul

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


def run_epoch(loader, model, criterion, optimizer=None):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            preds = torch.argmax(logits, dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, model, criterion, optimizer)
    val_loss, val_acc = run_epoch(val_loader, model, criterion, optimizer=None)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

# =====================
# 9) Loss & Accuracy grafikleri
# =====================
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(6, 4))
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Eğitim / Doğrulama Loss")
plt.legend()
plt.tight_layout()
loss_fig_path = FIG_DIR / "lstm_loss_curve.png"
plt.savefig(loss_fig_path, dpi=300)
plt.close()
print("LSTM loss grafiği kaydedildi:", loss_fig_path)

plt.figure(figsize=(6, 4))
plt.plot(epochs_range, train_accuracies, label="Train Acc")
plt.plot(epochs_range, val_accuracies, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("LSTM Eğitim / Doğrulama Accuracy")
plt.legend()
plt.tight_layout()
acc_fig_path = FIG_DIR / "lstm_accuracy_curve.png"
plt.savefig(acc_fig_path, dpi=300)
plt.close()
print("LSTM accuracy grafiği kaydedildi:", acc_fig_path)

# =====================
# 10) Test seti değerlendirmesi
# =====================
def predict_on_loader(loader, model):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


y_test_true, y_test_pred = predict_on_loader(test_loader, model)

print("\n--- LSTM Test Classification Report ---")
print(classification_report(y_test_true, y_test_pred, digits=4))

test_acc = accuracy_score(y_test_true, y_test_pred)
print("LSTM Test accuracy:", test_acc)

cm = confusion_matrix(y_test_true, y_test_pred)
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
plt.title("LSTM Confusion Matrix")
plt.tight_layout()
cm_path = FIG_DIR / "lstm_confusion_matrix.png"
plt.savefig(cm_path, dpi=300)
plt.close()
print("LSTM confusion matrix kaydedildi:", cm_path)
