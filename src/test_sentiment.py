# test_sentiment.py - quick test script for sentiment analysis

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from sentiment import SentimentAnalyzer
from response_adapter import build_prompt_instruction, shape_response

# test samples
SAMPLES = [
    ("I absolutely love this course, best decision ever!", "positive", False),
    ("Thanks, that answers my question perfectly.", "positive", False),
    ("Great customer service, super helpful!", "positive", False),
    ("What are the payment options for the bootcamp?", "neutral", False),
    ("Can I get the syllabus for this course?", "neutral", False),
    ("Do you support Mac OS for powerbi?", "neutral", False),
    ("This is the worst support experience I've ever had, I want a refund.", "negative", True),
    ("I've emailed three times and nobody has responded. Completely unacceptable.", "negative", True),
    ("This is a scam and total waste of money!", "negative", True),
    ("I am dissatisfied with the course quality.", "negative", False),
]


def test_sentiment_detection():
    print("Testing Sentiment Detection...")
    analyzer = SentimentAnalyzer(prefer_transformers=False)
    print("Engine:", analyzer.engine)

    correct = 0
    escalation_correct = 0

    for text, expected_sentiment, expected_escalate in SAMPLES:
        result = analyzer.analyze(text)
        sentiment_match = result["sentiment"] == expected_sentiment
        escalate_match = result["escalate"] == expected_escalate
        
        if sentiment_match:
            correct += 1
        if escalate_match:
            escalation_correct += 1

        status = "[PASS]" if (sentiment_match and escalate_match) else "[FAIL]"
        print(f"{status} [{result['sentiment']}] -> \"{text}\"")

    accuracy = (correct / len(SAMPLES)) * 100
    print(f"\nAccuracy: {correct}/{len(SAMPLES)} ({accuracy:.1f}%)\n")


def test_response_adaptation():
    print("Testing Response Adaptation...")
    
    pos_prompt = build_prompt_instruction("positive")
    neu_prompt = build_prompt_instruction("neutral")
    neg_prompt = build_prompt_instruction("negative")

    raw_answer = "We offer EMI options."
    shaped_pos = shape_response(raw_answer, {"sentiment": "positive", "escalate": False})
    shaped_neg = shape_response(raw_answer, {"sentiment": "negative", "escalate": True})

    print("[PASS] Prompt instructions and response shaping OK.\n")


def main():
    test_sentiment_detection()
    test_response_adaptation()
    print("All tests completed successfully!")


if __name__ == "__main__":
    main()
