# response_adapter.py - Shapes chatbot responses based on customer sentiment and escalation

# Distinct prompt tone instructions for the LLM
TONE_INSTRUCTIONS = {
    "positive": (
        "The customer is happy, enthusiastic, or appreciative. Respond warmly and encouragingly. "
        "Acknowledge their positive attitude, provide clear and helpful details, and reinforce their learning goals."
    ),
    "neutral": (
        "The customer is making an objective, informational inquiry. Provide a concise, clear, and direct factual response. "
        "Avoid unnecessary fluff or forced emotion, focusing strictly on resolving the question."
    ),
    "negative": (
        "The customer is frustrated, dissatisfied, or experiencing difficulties. Respond with deep empathy, patience, and a professional tone. "
        "Acknowledge their frustration first, apologize sincerely for the inconvenience, validate their experience, and offer a clear, supportive solution or next steps."
    ),
}

# Empathetic closers based on sentiment
CLOSERS = {
    "positive": (
        "\n\nWe are excited to support your learning journey! If you need any more details, feel free to ask."
    ),
    "neutral": (
        "\n\nIf you have any further questions or need additional assistance, please let us know."
    ),
    "negative": (
        "\n\nYour satisfaction is our priority. If this does not completely resolve your concern, please reply directly or contact support@nullclass.com so we can make this right."
    ),
}

ESCALATION_BANNER = (
    "\n\n---\n"
    "**[Senior Support Escalation Notice]**: This issue has been flagged for priority review by our human support team. "
    "A dedicated specialist has been notified and will follow up with you at your registered email within 2-4 hours. "
    "You can also reach our escalation desk directly at **support@nullclass.com**."
)


def build_prompt_instruction(sentiment):
    """Returns detailed tone instructions tailored to the customer's sentiment."""
    return TONE_INSTRUCTIONS.get(sentiment, TONE_INSTRUCTIONS["neutral"])


def shape_response(answer, sentiment_result):
    """
    Adapts the generated answer with sentiment-aligned openers, empathetic closers,
    and official human support escalation notices when flagged.
    """
    clean_ans = (answer or "").strip()
    if not clean_ans:
        clean_ans = (
            "I apologize, but I could not find verified information regarding your question in our official FAQ database. "
            "Please reach out to support@nullclass.com."
        )

    sentiment = sentiment_result.get("sentiment", "neutral").lower()
    escalate = sentiment_result.get("escalate", False)

    # Apply empathetic opener if not already present in the response
    opener = ""
    lower_ans = clean_ans.lower()
    if sentiment == "negative":
        if not any(w in lower_ans[:80] for w in ["sorry", "apologize", "understand your frustration", "sincerely"]):
            opener = "I sincerely apologize for the inconvenience and frustration you are experiencing. "
        closer = CLOSERS["negative"]
    elif sentiment == "positive":
        if not any(w in lower_ans[:60] for w in ["thank you", "great", "glad", "happy to help", "welcome"]):
            opener = "Thank you for reaching out! We're happy to assist you with this. "
        closer = CLOSERS["positive"]
    else:
        closer = CLOSERS["neutral"]

    # Assemble shaped response
    shaped = f"{opener}{clean_ans}{closer}"

    # Append priority escalation notice if flagged
    if escalate:
        shaped += ESCALATION_BANNER

    return shaped
