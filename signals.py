import re
import math
import pandas as pd

# ── Load ──────────────────────────────────────────────────
df = pd.read_csv("news_clean.csv")
print(f"Loaded {len(df)} articles")

# ── Urgency ───────────────────────────────────────────────
URGENCY_WORDS = {
    "breaking", "urgent", "alert", "exclusive", "crisis", "emergency",
    "disaster", "attack", "collapse", "shock", "scandal", "explosion",
    "outbreak", "warning", "critical", "deadly", "fatal", "killed",
    "murder", "war", "invasion", "protest", "arrested", "banned",
    "leaked", "exposed", "revealed", "just", "now", "latest", "today"
}

def urgency_score(text):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in URGENCY_WORDS)
    return min(hits / (len(words) * 0.05), 1.0)

# ── Emotion ───────────────────────────────────────────────
POSITIVE_WORDS = {
    "amazing", "incredible", "victory", "triumph", "hero", "saves",
    "record", "historic", "breakthrough", "best", "first", "launch",
    "wins", "celebrates", "success", "boom", "soars"
}
NEGATIVE_WORDS = {
    "tragedy", "death", "violence", "fear", "threat", "danger",
    "fail", "worst", "crash", "loss", "suffers", "victim",
    "accused", "guilty", "arrested", "fraud", "lie", "hoax"
}

def emotion_score(text):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    return min((pos + neg * 1.3) / (len(words) * 0.04), 1.0)

# ── Lexical Diversity ─────────────────────────────────────
def lexical_diversity(text):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)

# ── Readability ───────────────────────────────────────────
def count_syllables(word):
    word = word.lower()
    vowels = "aeiouy"
    count, prev_vowel = 0, False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e"):
        count = max(count - 1, 1)
    return max(count, 1)

def readability_score(text):
    words     = re.findall(r'\b[a-z]+\b', text.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    num_words     = max(len(words), 1)
    num_sentences = max(len(sentences), 1)
    syllables     = sum(count_syllables(w) for w in words)
    flesch = 206.835 - (1.015 * num_words / num_sentences) - (84.6 * syllables / num_words)
    return max(0.0, min(flesch / 100, 1.0))

# ── Length ────────────────────────────────────────────────
def length_score(text):
    return math.exp(-((len(text) - 600) ** 2) / (2 * 500 ** 2))

# ── Subjectivity ──────────────────────────────────────────
SUBJECTIVE = {
    "i", "we", "our", "my", "your", "you", "believe", "think",
    "feel", "opinion", "must", "should", "would"
}

def subjectivity_score(text):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0.0
    return min(sum(1 for w in words if w in SUBJECTIVE) / len(words) * 10, 1.0)

# ── Question Hook ─────────────────────────────────────────
QUESTION_STARTERS = {"why","how","what","who","when","will","can","is","are","does"}

def question_hook(title):
    title_lower = title.lower()
    return 1.0 if ("?" in title or any(title_lower.startswith(w) for w in QUESTION_STARTERS)) else 0.0

# ── Named Entities ────────────────────────────────────────
def named_entity_score(text):
    words      = re.findall(r'\b\w+\b', text)
    cap_tokens = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
    if not words:
        return 0.0
    return min(len(cap_tokens) / len(words) * 5, 1.0)

# ── Weights ───────────────────────────────────────────────
WEIGHTS = {
    "urgency":           0.22,
    "emotion":           0.20,
    "lexical_diversity": 0.12,
    "readability":       0.10,
    "length":            0.08,
    "subjectivity":      0.08,
    "question_hook":     0.10,
    "named_entities":    0.10,
}

# ── Combine all signals ───────────────────────────────────
def compute_signals(title, description):
    text = f"{title} {description}"
    signals = {
        "urgency":           urgency_score(text),
        "emotion":           emotion_score(text),
        "lexical_diversity": lexical_diversity(text),
        "readability":       readability_score(text),
        "length":            length_score(text),
        "subjectivity":      subjectivity_score(text),
        "question_hook":     question_hook(title),
        "named_entities":    named_entity_score(text),
    }
    signals["pseudo_label"] = round(sum(WEIGHTS[k] * signals[k] for k in WEIGHTS), 4)
    return signals

# ── Run on all articles ───────────────────────────────────
print("Computing signals... (this takes 2-4 minutes)")

signal_rows = []
for i, row in df.iterrows():
    signal_rows.append(compute_signals(row['Title'], row['Description']))
    if i % 10000 == 0:
        print(f"  {i:,} / {len(df):,} done...")

signals_df = pd.DataFrame(signal_rows)
df_final   = pd.concat([df, signals_df], axis=1)

df_final.to_csv("news_signals.csv", index=False)

print(f"\nDone! Saved news_signals.csv")
print(f"\nSignal averages across all articles:")
print(signals_df.mean().round(3))