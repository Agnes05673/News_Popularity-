import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

print("Libraries loaded")
print(f"PyTorch version: {torch.__version__}")
# ── Load cleaned data ─────────────────────────────────────
df = pd.read_csv("news_signals.csv").head(10000)
print(f"Using {len(df)} articles (subset for memory efficiency)")


# ── Load DistilBERT ───────────────────────────────────────
print("\nLoading DistilBERT... (first time downloads ~250MB)")
MODEL_NAME = "distilbert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
print("DistilBERT loaded successfully")
# ── Extract embeddings for all articles ───────────────────
BATCH_SIZE = 16
TOTAL = len(df)

print(f"\nExtracting embeddings for all {TOTAL} articles...")
print("This will take 20-40 minutes on CPU. Please wait...")

# Pre-create the output file so we write directly to disk
embeddings_file = np.lib.format.open_memmap(
    "embeddings.npy",
    mode="w+",
    dtype="float32",
    shape=(TOTAL, 768)
)

for batch_num, start in enumerate(range(0, TOTAL, BATCH_SIZE)):
    end   = min(start + BATCH_SIZE, TOTAL)
    batch = df.iloc[start:end]

    texts = [
        f"{row['Title']} [SEP] {row['Description']}"
        for _, row in batch.iterrows()
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    cls_vectors = outputs.last_hidden_state[:, 0, :].numpy()
    
    # write directly to disk — no RAM buildup
    embeddings_file[start:end] = cls_vectors

    if batch_num % 50 == 0:
        print(f"  {end:,} / {TOTAL:,} done...")

# flush to disk
embeddings_file.flush()
print(f"\nDone! Saved embeddings.npy")
print(f"Shape: {embeddings_file.shape}")
print(f"Expected: (216900, 768)")

# ── Load embeddings and pseudo labels ─────────────────────
embeddings   = np.load("embeddings.npy")
pseudo_labels = df['pseudo_label'].values.astype('float32')

print(f"Embeddings shape: {embeddings.shape}")
print(f"Labels shape: {pseudo_labels.shape}")

# ── Dataset class ─────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels,     dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ── Regression head ───────────────────────────────────────
class PopularityRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze()

# ── Training setup ────────────────────────────────────────
dataset    = NewsDataset(embeddings, pseudo_labels)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

regressor  = PopularityRegressor()
optimizer  = torch.optim.Adam(regressor.parameters(), lr=0.001)
criterion  = nn.MSELoss()

# ── Train ─────────────────────────────────────────────────
print("\nTraining regression head...")
EPOCHS = 10

for epoch in range(EPOCHS):
    regressor.train()
    total_loss = 0

    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        predictions = regressor(X_batch)
        loss        = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

# ── Save model ────────────────────────────────────────────
torch.save(regressor.state_dict(), "regressor.pth")
print("\nDone! Saved regressor.pth")