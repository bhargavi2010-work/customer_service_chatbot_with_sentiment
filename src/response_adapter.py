# response_adapter.py - shapes chatbot response based on sentiment

# prompt tone instructions for the LLM
TONE_INSTRUCTIONS = {
    "positive": "The customer sounds happy or satisfied. Answer clearly and politely.",
    "neutral": "Answer clearly, politely, and to the point.",
    "negative": "The customer sounds frustrated. Acknowledge their issue politely before answering.",
}


def build_prompt_instruction(sentiment):
    return TONE_INSTRUCTIONS.get(sentiment, TONE_INSTRUCTIONS["neutral"])


def shape_response(answer, sentiment_result):
    # keep response clean and accurate from document context
    return answer.strip()
