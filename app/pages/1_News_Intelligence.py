# app/pages/1_News_Intelligence.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from scorer import score_article, tier

st.set_page_config(page_title="News Intelligence", page_icon="🔍", layout="wide")
st.title("🔍 News Intelligence")
st.markdown("Enter a news article title and description to get its predicted popularity score.")
st.markdown("---")

title       = st.text_input("📰 Article Title", placeholder="e.g. BREAKING: Earthquake strikes Tokyo")
description = st.text_area("📝 Article Description", placeholder="Enter the article summary or description here...", height=150)

if st.button("🚀 Analyse Article", type="primary"):
    if not title.strip():
        st.warning("Please enter a title.")
    else:
        with st.spinner("Analysing article..."):
            score, signals = score_article(title, description)
            label, color   = tier(score)

        st.markdown("---")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"### Popularity Score")
            st.markdown(f"<h1 style='color:{color};font-size:64px'>{score}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)
            st.caption("Score out of 100")

        with col2:
            st.markdown("### Signal Breakdown")
            for signal, value in signals.items():
                st.markdown(f"**{signal}**")
                st.progress(value, text=f"{value:.3f}")

        st.markdown("---")
        st.markdown("### 💡 Explanation")

        top    = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:3]
        bottom = sorted(signals.items(), key=lambda x: x[1])[:2]

        top_names    = ", ".join([k for k, _ in top])
        bottom_names = ", ".join([k for k, _ in bottom])

        st.markdown(f"""
        This article scored **{score}/100** ({label}).
        
        **Strongest signals:** {top_names} — these indicate high reader engagement potential.
        
        **Weakest signals:** {bottom_names} — these slightly limit the predicted reach.
        """)

        if signals["Urgency"] > 0.3:
            st.success("✅ Strong urgency language detected — time-sensitive content performs well.")
        if signals["Question Hook"] > 0.5:
            st.success("✅ Question-style title detected — curiosity gaps drive clicks.")
        if signals["Emotion"] > 0.3:
            st.success("✅ High emotional intensity — emotional content gets shared more.")
        if score < 30:
            st.info("💡 Tip: Adding urgency language or a question hook to the title could improve predicted reach.")
            