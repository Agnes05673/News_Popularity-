# app/scorer.py
import re
import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import streamlit as st

URGENCY_WORDS  = {"breaking","urgent","alert","crisis","emergency","disaster",
                  "attack","collapse","shock","scandal","deadly","fatal","killed",
                  "murder","war","arrested","leaked","exposed","revealed","today"}
POSITIVE_WORDS = {"amazing","incredible","victory","triumph","hero","record",
                  "historic","breakthrough","best","first","success","boom"}
NEGATIVE_WORDS = {"tragedy","death","violence","fear","threat","danger","fail",
                  "worst","crash","loss","victim","accused","guilty","fraud"}
SUBJECTIVE     = {"we","our","believe","think","feel","must","should","would"}
QUESTION_START = {"why","how","what","who","when","will","can","is","are","does"}

WEIGHTS = {
    "Urgency":           0.30,
    "Emotion":           0.22,
    "Lexical Diversity": 0.12,
    "Length":            0.08,
    "Subjectivity":      0.08,
    "Question Hook":     0.05,
    "Named Entities":    0.10,
}

class PopularityRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),    nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x).squeeze()

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert      = AutoModel.from_pretrained("distilbert-base-uncased")
    bert.eval()
    regressor = PopularityRegressor()
    regressor.load_state_dict(torch.load(
        "regressor.pth", map_location="cpu", weights_only=True))
    regressor.eval()
    return tokenizer, bert, regressor

def get_signals(title, description):
    text  = f"{title} {description}"
    words = re.findall(r'\b[a-z]+\b', text.lower())
    n     = max(len(words), 1)
    return {
        "Urgency":           round(min(sum(1 for w in words if w in URGENCY_WORDS) / (n * 0.05), 1.0), 3),
        "Emotion":           round(min((sum(1 for w in words if w in POSITIVE_WORDS) + sum(1 for w in words if w in NEGATIVE_WORDS) * 1.3) / (n * 0.04), 1.0), 3),
        "Lexical Diversity": round(len(set(words)) / n, 3),
        "Length":            round(math.exp(-((len(text) - 600) ** 2) / (2 * 500 ** 2)), 3),
        "Subjectivity":      round(min(sum(1 for w in words if w in SUBJECTIVE) / n * 10, 1.0), 3),
        "Question Hook":     round(1.0 if ("?" in title or any(title.lower().startswith(w) for w in QUESTION_START)) else 0.0, 3),
        "Named Entities":    round(min(len(re.findall(r'\b[A-Z][a-zA-Z]+\b', text)) / n * 5, 1.0), 3),
    }

def score_article(title, description):
    tokenizer, bert, regressor = load_models()
    text   = f"{title} [SEP] {description}"
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        out       = bert(**inputs)
        emb       = out.last_hidden_state[:, 0, :].squeeze().unsqueeze(0)
        emb_score = regressor(emb).item()

    signals      = get_signals(title, description)
    signal_score = sum(signals[k] * WEIGHTS[k] for k in WEIGHTS)
    final        = (0.70 * signal_score) + (0.30 * emb_score)
    stretched    = 1 / (1 + math.exp(-10 * (final - 0.4)))
    score        = round(stretched * 100, 1)
    return score, signals

def tier(score):
    if score >= 80:   return "🔥 Viral",    "#FF4B4B"
    elif score >= 65: return "📈 High",     "#FF8C00"
    elif score >= 45: return "📊 Moderate", "#FBBF24"
    elif score >= 25: return "📉 Low",      "#94A3B8"
    else:             return "❄️ Minimal",  "#64748B"