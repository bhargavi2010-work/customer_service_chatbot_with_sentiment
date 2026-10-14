# main.py - streamlit frontend for customer service chatbot

import streamlit as st
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from langchain_helper import get_response, create_vector_db

st.set_page_config(
    page_title="Customer Service Chatbot",
    page_icon="🤖",
    layout="wide"
)

# initialize sentiment history in session state
if "sentiment_history" not in st.session_state:
    st.session_state["sentiment_history"] = []

st.title("CUSTOMER SERVICE CHATBOT 🤖")

SENTIMENT_ICONS = {
    "positive": "😊",
    "neutral": "😐",
    "negative": "😡",
}

SENTIMENT_COLORS = {
    "positive": "#4CAF50",
    "neutral": "#FFC107",
    "negative": "#E53935",
}

# Main Layout: Left Chat Area & Far-Right Analytics Sidebar Column
left_col, right_col = st.columns([3.8, 1.2], gap="large")

with left_col:
    btn = st.button("Create Knowledgebase")
    if btn:
        with st.spinner("Creating vector database..."):
            create_vector_db()
            st.success("Knowledgebase created successfully!")

    question = st.text_input("Question: ")

    if question:
        with st.spinner("Generating answer..."):
            try:
                result = get_response(question)
                st.session_state["last_result"] = result

                # append sentiment to session state history
                sentiment = result["sentiment"].lower()
                st.session_state["sentiment_history"].append(sentiment)

                st.header("Answer")
                st.write(result["answer"])

                # If escalated, show prominent human escalation action banner
                if result.get("escalate"):
                    st.error(
                        "🚨 **Action Required: Ticket Escalated to Human Support**\n\n"
                        "This interaction has been flagged for priority intervention. A senior support specialist "
                        "has been notified and will follow up with the user. You can also reach human dispatch directly at **support@nullclass.com**."
                    )

                # Sentiment Analysis Card Box
                with st.container(border=True):
                    st.markdown("### 📊 Sentiment & Tone Analysis")
                    
                    s_color = "#dc2626" if sentiment == "negative" else ("#16a34a" if sentiment == "positive" else "#ca8a04")
                    s_bg = "#fee2e2" if sentiment == "negative" else ("#dcfce7" if sentiment == "positive" else "#fef9c3")
                    
                    esc_val = "Yes (Priority Queue)" if result["escalate"] else "No"
                    esc_color = "#dc2626" if result["escalate"] else "#16a34a"
                    esc_bg = "#fee2e2" if result["escalate"] else "#dcfce7"

                    reason_text = "High customer frustration or urgent escalation intent detected" if result["escalate"] else ("Customer expressed satisfaction or enthusiasm" if sentiment == "positive" else "Standard informational inquiry")

                    st.markdown(f"**Detected Sentiment:** &nbsp;&nbsp; <span style='background-color:{s_bg}; color:{s_color}; padding:4px 14px; border-radius:12px; font-weight:bold;'>{sentiment.capitalize()}</span>", unsafe_allow_html=True)
                    st.markdown(f"<div style='margin-top:10px;'><b>Confidence Score:</b> &nbsp;&nbsp; <span style='background-color:#dbeafe; color:#2563eb; padding:4px 14px; border-radius:12px; font-weight:bold;'>{result['confidence']:.1%}</span></div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='margin-top:10px;'><b>Escalation Required:</b> &nbsp;&nbsp; <span style='background-color:{esc_bg}; color:{esc_color}; padding:4px 14px; border-radius:12px; font-weight:bold;'>{esc_val}</span></div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='margin-top:10px;'><b>Reason:</b> &nbsp;&nbsp; {reason_text}</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")

# Right Column: Dashboard Cards (ONLY Analytics Dashboard & Sentiment History)
with right_col:
    if st.session_state.get("last_result"):
        last = st.session_state["last_result"]
        s_type = last["sentiment"].lower()

        # Card 1: Current Sentiment
        with st.container(border=True):
            st.markdown("#### Current Sentiment")
            c1, c2 = st.columns([1, 2])
            with c1:
                icon = SENTIMENT_ICONS.get(s_type, "😐")
                color = SENTIMENT_COLORS.get(s_type, "#FFC107")
                st.markdown(f"<h1 style='text-align: center; color: {color}; font-size: 2.5rem; margin: 0;'>{icon}</h1>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<h4 style='color: {color}; margin: 0;'>{s_type.capitalize()}</h4>", unsafe_allow_html=True)
                st.write(f"**Confidence:** {last['confidence']:.1%}")
            st.progress(float(last["confidence"]))

        # Card 2: Sentiment History
        with st.container(border=True):
            st.markdown("#### Sentiment History")
            history = st.session_state["sentiment_history"]
            total = len(history)

            pos_count = history.count("positive")
            neu_count = history.count("neutral")
            neg_count = history.count("negative")

            pos_pct = (pos_count / total) * 100 if total > 0 else 0
            neu_pct = (neu_count / total) * 100 if total > 0 else 0
            neg_pct = (neg_count / total) * 100 if total > 0 else 0

            fig, ax = plt.subplots(figsize=(0.9, 0.9))
            labels, sizes, colors = [], [], []
            if pos_count > 0:
                labels.append("Positive"); sizes.append(pos_count); colors.append("#4CAF50")
            if neu_count > 0:
                labels.append("Neutral"); sizes.append(neu_count); colors.append("#FFC107")
            if neg_count > 0:
                labels.append("Negative"); sizes.append(neg_count); colors.append("#E53935")

            if not sizes:
                sizes = [1]
                colors = ["#CCCCCC"]

            ax.pie(sizes, colors=colors, startangle=90)
            ax.axis("equal")
            fig.patch.set_alpha(0.0)
            st.pyplot(fig)

            st.markdown(f"<p style='color: #4CAF50; font-weight: bold; margin:0;'>🟢 Positive: {pos_pct:.0f}%</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #FFC107; font-weight: bold; margin:0;'>🟡 Neutral: {neu_pct:.0f}%</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #E53935; font-weight: bold; margin:0;'>🔴 Negative: {neg_pct:.0f}%</p>", unsafe_allow_html=True)
