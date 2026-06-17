import re
import html
import pandas as pd

# ── 1. Load ──────────────────────────────────────────────
df = pd.read_csv("News_dataset .csv")
print(f"Loaded: {df.shape}")
print(f"Nulls:\n{df.isnull().sum()}")

# ── 2. Fill nulls ─────────────────────────────────────────
df['Description'] = df['Description'].fillna("")

# ── 3. Clean function ─────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)           # &quot; → "
    text = re.sub(r'<[^>]+>', ' ', text) # remove HTML tags
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'\s+', ' ', text)     # collapse spaces
    return text.strip()

# ── 4. Apply ──────────────────────────────────────────────
df['Title']       = df['Title'].apply(clean_text)
df['Description'] = df['Description'].apply(clean_text)

# ── 5. Combined text column ───────────────────────────────
df['text'] = df['Title'] + " [SEP] " + df['Description']

# ── 6. Save ───────────────────────────────────────────────
df.to_csv("news_clean.csv", index=False)
print(f"\nDone! Saved news_clean.csv with {len(df)} rows")
print(f"\nSample:")
print(df[['Title','Description']].head(3))
