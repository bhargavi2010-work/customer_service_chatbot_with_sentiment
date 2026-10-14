# test_sentiment.py - Unit test suite for sentiment detection, edge cases, and response shaping

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from sentiment import SentimentAnalyzer
from response_adapter import build_prompt_instruction, shape_response, CLOSERS, ESCALATION_BANNER

# Benchmark test cases including mandatory reviewer edge cases
TEST_CASES = [
    # Positive satisfaction and appreciation
    ("I absolutely love this course, best decision ever!", "positive", False),
    ("Thanks, that answers my question perfectly.", "positive", False),
    ("Great customer service, super helpful!", "positive", False),

    # Informational FAQ queries (must be NEUTRAL, no escalation)
    ("What are the payment options for the bootcamp?", "neutral", False),
    ("Can I get the syllabus for this course?", "neutral", False),
    ("Do you support Mac OS for powerbi?", "neutral", False),
    ("I want to book a demo class", "neutral", False),
    ("What is your refund policy?", "neutral", False),
    ("tissue", "neutral", False),

    # Technical issue (Neutral inquiry, NOT escalated by 'sue' in 'issue')
    ("I have an issue logging into my account", "neutral", False),

    # Technical complaints / dissatisfaction (Negative sentiment, but not urgent escalation)
    ("The course video is broken and nothing works", "negative", False),
    ("I am dissatisfied with the course quality.", "negative", False),
    ("I'm not interested in this course anymore", "negative", False),

    # Severe complaints & urgent refund demands (Must trigger negative AND escalate=True)
    ("This is the worst support experience I've ever had, I want a refund.", "negative", True),
    ("I've emailed three times and nobody has responded. Completely unacceptable.", "negative", True),
    ("This is a scam and total waste of money!", "negative", True),
    ("I want a refund right now!", "negative", True),
]


def test_sentiment_detection():
    print("=" * 60)
    print("     TESTING MULTI-CLASS SENTIMENT DETECTION & ESCALATION    ")
    print("=" * 60)

    analyzer = SentimentAnalyzer(prefer_transformers=False)
    print(f"Active Engine: {analyzer.engine}\n")

    passed = 0
    total = len(TEST_CASES)

    for text, expected_sentiment, expected_escalate in TEST_CASES:
        res = analyzer.analyze(text)
        sent = res["sentiment"]
        conf = res["confidence"]
        esc = res["escalate"]

        sent_ok = sent == expected_sentiment
        esc_ok = esc == expected_escalate

        status = "PASS" if (sent_ok and esc_ok) else "FAIL"
        if status == "PASS":
            passed += 1

        print(f"[{status}] [{sent:8}] [Escalate: {str(esc):5}] (Conf: {conf:.3f}) -> \"{text}\"")
        if not sent_ok or not esc_ok:
            print(f"   --> EXPECTED: [{expected_sentiment}] [Escalate: {expected_escalate}]")

    acc = (passed / total) * 100
    print("\n" + "=" * 60)
    print(f"Detection Accuracy: {passed}/{total} ({acc:.1f}%)")
    print("=" * 60 + "\n")
    assert acc >= 90.0, f"Accuracy below threshold: {acc:.1f}%"


def test_response_adaptation():
    print("=" * 60)
    print("          TESTING RESPONSE ADAPTATION & SHAPING          ")
    print("=" * 60)

    raw_answer = "We offer flexible EMI options on all courses."

    # Test Positive Adaptation
    pos_res = shape_response(raw_answer, {"sentiment": "positive", "escalate": False})
    print("[Positive Shaped Answer]:\n", pos_res)
    assert raw_answer in pos_res
    assert CLOSERS["positive"] in pos_res
    assert "Thank you" in pos_res or "happy" in pos_res
    print("-> Positive adaptation verified.\n")

    # Test Neutral Adaptation
    neu_res = shape_response(raw_answer, {"sentiment": "neutral", "escalate": False})
    print("[Neutral Shaped Answer]:\n", neu_res)
    assert raw_answer in neu_res
    assert CLOSERS["neutral"] in neu_res
    print("-> Neutral adaptation verified.\n")

    # Test Negative with Escalation Adaptation
    neg_res = shape_response(raw_answer, {"sentiment": "negative", "escalate": True})
    print("[Negative + Escalated Shaped Answer]:\n", neg_res)
    assert raw_answer in neg_res
    assert "apologize" in neg_res.lower()
    assert CLOSERS["negative"] in neg_res
    assert "Senior Support Escalation Notice" in neg_res
    print("-> Negative + Escalation adaptation verified.\n")

    print("=" * 60)
    print("Response adaptation verified successfully!")
    print("=" * 60 + "\n")


def test_dynamic_confidence():
    """Verify confidence scores are dynamic model outputs, not static constants."""
    analyzer = SentimentAnalyzer(prefer_transformers=False)
    res1 = analyzer.analyze("I love this course!")
    res2 = analyzer.analyze("Good class.")
    res3 = analyzer.analyze("The video is broken and audio fails.")

    confs = [res1["confidence"], res2["confidence"], res3["confidence"]]
    print(f"Dynamic confidences across inputs: {confs}")
    assert len(set(confs)) > 1, "Confidences should vary dynamically based on the model!"


def main():
    test_sentiment_detection()
    test_response_adaptation()
    test_dynamic_confidence()
    print("All unit tests PASSED successfully!")


if __name__ == "__main__":
    main()
