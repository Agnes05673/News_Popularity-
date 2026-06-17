import re
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# ── Load DistilBERT ───────────────────────────────────────
print("Loading DistilBERT...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert      = AutoModel.from_pretrained("distilbert-base-uncased")
bert.eval()

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

regressor = PopularityRegressor()
regressor.load_state_dict(torch.load("regressor.pth", weights_only=True))
regressor.eval()
print("Model loaded\n")

# ── Proxy signals (same as Step 3) ───────────────────────
URGENCY_WORDS  = {"breaking","urgent","alert","crisis","emergency","disaster",
                  "attack","collapse","shock","scandal","deadly","fatal","killed",
                  "murder","war","arrested","leaked","exposed","revealed","today"}
POSITIVE_WORDS = {"amazing","incredible","victory","triumph","hero","record",
                  "historic","breakthrough","best","first","success","boom"}
NEGATIVE_WORDS = {"tragedy","death","violence","fear","threat","danger","fail",
                  "worst","crash","loss","victim","accused","guilty","fraud"}
SUBJECTIVE     = {"we","our","believe","think","feel","must","should","would"}
QUESTION_START = {"why","how","what","who","when","will","can","is","are","does"}

def get_signals(title, description):
    text  = f"{title} {description}"
    words = re.findall(r'\b[a-z]+\b', text.lower())
    n     = max(len(words), 1)

    urgency  = min(sum(1 for w in words if w in URGENCY_WORDS)  / (n * 0.05), 1.0)
    emotion  = min((sum(1 for w in words if w in POSITIVE_WORDS) +
                    sum(1 for w in words if w in NEGATIVE_WORDS) * 1.3) / (n * 0.04), 1.0)
    lexical  = len(set(words)) / n
    length   = math.exp(-((len(text) - 600) ** 2) / (2 * 500 ** 2))
    subject  = min(sum(1 for w in words if w in SUBJECTIVE) / n * 10, 1.0)
    question = 1.0 if ("?" in title or any(title.lower().startswith(w) for w in QUESTION_START)) else 0.0
    entities = min(len(re.findall(r'\b[A-Z][a-zA-Z]+\b', text)) / n * 5, 1.0)

    return {
        "Urgency":            round(urgency,  3),
        "Emotion":            round(emotion,  3),
        "Lexical Diversity":  round(lexical,  3),
        "Length":             round(length,   3),
        "Subjectivity":       round(subject,  3),
        "Question Hook":      round(question, 3),
        "Named Entities":     round(entities, 3),
    }

def get_embedding(title, description):
    text   = f"{title} [SEP] {description}"
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        out = bert(**inputs)
    return out.last_hidden_state[:, 0, :].squeeze()

WEIGHTS = {
    "Urgency":           0.30,
    "Emotion":           0.22,
    "Lexical Diversity": 0.12,
    "Length":            0.08,
    "Subjectivity":      0.08,
    "Question Hook":     0.05,
    "Named Entities":    0.10,
}

def score_article(title, description):
    emb = get_embedding(title, description).unsqueeze(0)
    with torch.no_grad():
        emb_score = regressor(emb).item()

    signals = get_signals(title, description)

    signal_score = (
        signals["Urgency"]           * WEIGHTS["Urgency"]           +
        signals["Emotion"]           * WEIGHTS["Emotion"]           +
        signals["Lexical Diversity"] * WEIGHTS["Lexical Diversity"] +
        signals["Length"]            * WEIGHTS["Length"]            +
        signals["Subjectivity"]      * WEIGHTS["Subjectivity"]      +
        signals["Question Hook"]     * WEIGHTS["Question Hook"]     +
        signals["Named Entities"]    * WEIGHTS["Named Entities"]
    )

    final     = (0.70 * signal_score) + (0.30 * emb_score)
    stretched = 1 / (1 + math.exp(-10 * (final - 0.4)))
    score     = round(stretched * 100, 1)

    return score, signals

def tier(score):
    if score >= 80: return "🔥 Viral"
    elif score >= 65: return "📈 High"
    elif score >= 45: return "📊 Moderate"
    elif score >= 25: return "📉 Low"
    else: return "❄️ Minimal"

# ── Case studies ──────────────────────────────────────────
articles = [
    {
        "title": "BREAKING: Major earthquake strikes Tokyo, thousands feared dead",
        "description": "A catastrophic 8.1 magnitude earthquake has struck central Tokyo killing hundreds and leaving thousands trapped under collapsed buildings. Emergency services are overwhelmed."
    },
    {
        "title": "Scientists discover potential cure for cancer in breakthrough study",
        "description": "Researchers at Harvard Medical School have identified a molecule that successfully eliminates tumor cells in 94% of cases. Clinical trials begin next month."
    },
    {
        "title": "Why is the US economy heading toward recession?",
        "description": "Economists warn that rising inflation, falling consumer confidence, and Federal Reserve rate hikes are pushing the United States toward its worst recession in decades."
    },
    {
        "title": "City council approves new parking regulations",
        "description": "The municipal council voted to update parking rules in the downtown area. New signs will be installed over the coming weeks."
    },
    {
        "title": "Annual flower show opens at botanical gardens this weekend",
        "description": "Visitors can enjoy over 200 varieties of flowers from around the world at the city botanical gardens. The event runs Saturday and Sunday."
    },
]

print("=" * 60)
print("CASE STUDIES — Popularity Score Breakdown")
print("=" * 60)

results = []
for art in articles:
    score, signals = score_article(art['title'], art['description'])
    results.append({"title": art['title'], "score": score})

    print(f"\nTitle  : {art['title']}")
    print(f"Score  : {score}/100  {tier(score)}")
    print(f"Signals:")
    for k, v in signals.items():
        bar = "█" * int(v * 20)
        print(f"  {k:<20} {v:.3f}  {bar}")

# ── Ranking comparison ────────────────────────────────────
print("\n" + "=" * 60)
print("RANKING — Most to Least Popular")
print("=" * 60)
ranked = sorted(results, key=lambda x: x['score'], reverse=True)
for i, r in enumerate(ranked, 1):
    print(f"  #{i}  {r['score']:>5}/100  {tier(r['score'])}  {r['title'][:55]}")
