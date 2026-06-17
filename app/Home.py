# app/Home.py
import streamlit as st

st.set_page_config(
    page_title="News Popularity Intelligence System",
    page_icon="📰",
    layout="wide"
)

st.title("📰 News Popularity Intelligence System")
st.subheader("Predicting news article popularity from text alone — no labels required")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 What is this system?")
    st.markdown("""
    This system predicts how popular a news article is likely to be 
    **before it is published** — using only its title and description.
    
    No clicks. No shares. No historical data. Just text.
    """)

    st.markdown("### ❓ Why are popularity labels unavailable?")
    st.markdown("""
    When a news article is first published it has **zero engagement history**.
    
    - Clicks accumulate over hours
    - Shares happen after readers react  
    - Impressions depend on platform distribution
    
    This is called the **cold-start problem** — and it's exactly what 
    this system solves using AI.
    """)

with col2:
    st.markdown("### ⚙️ How does it work?")
    st.markdown("""
    The system combines two approaches:
    
    **1. Proxy Signal Engineering**
    Eight measurable text features that correlate with engagement:
    - Urgency language
    - Emotional intensity  
    - Lexical diversity
    - Readability
    - Content length
    - Subjectivity
    - Question hooks
    - Named entity density
    
    **2. DistilBERT Transformer**
    A pretrained language model that reads the article and produces 
    a 768-dimensional semantic embedding — capturing deep meaning, 
    tone, and context that simple word counting cannot.
    """)

st.markdown("---")
st.markdown("### 🏗️ System Architecture")

st.code("""
News Article (Title + Description)
            │
            ├──→ 8 Proxy Signals ──→ Weighted Score (70%)
            │      urgency, emotion,
            │      readability...
            │
            └──→ DistilBERT Encoder
                      │
                 CLS Embedding (768-dim)
                      │
                 Regression Head ──→ Embedding Score (30%)
                 (768→256→64→1)
                      │
                 ─────┴─────
                 Final Popularity Score (0-100)
""", language="text")

st.markdown("---")
st.info("👈 Use the sidebar to navigate to **News Intelligence** to score an article, or **Model Reasoning** to understand how the system works.")