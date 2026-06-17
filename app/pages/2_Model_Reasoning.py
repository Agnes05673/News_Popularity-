# app/pages/2_Model_Reasoning.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from scorer import score_article, tier

st.set_page_config(page_title="Model Reasoning", page_icon="🧠", layout="wide")
st.title("🧠 Model Reasoning")
st.markdown("Understand how and why the system scores articles the way it does.")
st.markdown("---")

# ── Scoring logic explanation ─────────────────────────────
st.markdown("### ⚙️ How the Score is Calculated")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Step 1 — Proxy Signals (70% of score)**
    
    Eight signals are computed from raw text:
    
    | Signal | Weight | Measures |
    |---|---|---|
    | Urgency | 30% | Crisis/breaking keywords |
    | Emotion | 22% | Charged positive/negative words |
    | Lexical Diversity | 12% | Vocabulary richness |
    | Named Entities | 10% | People, places, organisations |
    | Question Hook | 5% | Curiosity gap in title |
    | Length | 8% | Content depth |
    | Subjectivity | 8% | Opinion/perspective cues |
    """)

with col2:
    st.markdown("""
    **Step 2 — DistilBERT Embedding (30% of score)**
    
    The article is passed through DistilBERT, a pretrained 
    Transformer model with 66 million parameters.
    
    It produces a **768-dimensional CLS vector** — a semantic 
    fingerprint of the article's meaning, tone, and context.
    
    A lightweight regression head (768→256→64→1) was trained 
    on 10,000 articles to map this vector to a popularity score.
    
    **Final formula:**
   score = 70% × signal_score + 30% × embedding_score
                Applied through a sigmoid stretch to spread scores across 0–100.
    """)

st.markdown("---")

# ── Live comparison ───────────────────────────────────────
st.markdown("### 🔬 Live Article Comparison")
st.markdown("Compare two articles side by side to see how the system differentiates them.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Article A**")
    title_a = st.text_input("Title A", value="BREAKING: Major earthquake strikes Tokyo, thousands feared dead")
    desc_a  = st.text_area("Description A", value="A catastrophic 8.1 magnitude earthquake has struck central Tokyo killing hundreds. Emergency services are overwhelmed as buildings collapse.", height=100)

with col2:
    st.markdown("**Article B**")
    title_b = st.text_input("Title B", value="City council approves new parking regulations")
    desc_b  = st.text_area("Description B", value="The municipal council voted to update parking rules in the downtown area. New signs will be installed over the coming weeks.", height=100)

if st.button("⚖️ Compare Articles", type="primary"):
    with st.spinner("Analysing both articles..."):
        score_a, signals_a = score_article(title_a, desc_a)
        score_b, signals_b = score_article(title_b, desc_b)

    label_a, color_a = tier(score_a)
    label_b, color_b = tier(score_b)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<h2 style='color:{color_a}'>{score_a}/100 {label_a}</h2>", unsafe_allow_html=True)
        for signal, value in signals_a.items():
            st.markdown(f"**{signal}**")
            st.progress(value, text=f"{value:.3f}")

    with col2:
        st.markdown(f"<h2 style='color:{color_b}'>{score_b}/100 {label_b}</h2>", unsafe_allow_html=True)
        for signal, value in signals_b.items():
            st.markdown(f"**{signal}**")
            st.progress(value, text=f"{value:.3f}")

    st.markdown("---")
    winner = "Article A" if score_a > score_b else "Article B"
    diff   = abs(score_a - score_b)
    st.success(f"**{winner}** is predicted to be more popular by **{diff:.1f} points**.")

st.markdown("---")
st.markdown("### 📚 Why This Approach Works")
st.markdown("""
Traditional popularity prediction requires labelled data — clicks, shares, impressions. 
This system uses **weak supervision**: measurable text properties that correlate with engagement 
based on journalism research and media analytics findings.

The combination of proxy signals and deep Transformer embeddings allows the system to:
- Capture surface-level urgency and emotion (signals)
- Understand deeper semantic meaning and context (DistilBERT)
- Produce explainable, justified scores (signal breakdown)

This is a legitimate research technique used in industry when labelled data is unavailable.
""")