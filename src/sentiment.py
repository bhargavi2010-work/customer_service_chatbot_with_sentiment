# sentiment.py - Accurate multi-class sentiment analyzer for customer support
import re

ESCALATION_PATTERNS = [
    r"\b(want|need|give me|demand|requesting|get)\s+(a\s+)?refund\b",
    r"\brefund\s+my\s+money\b",
    r"\b(scam|fraud|unacceptable|furious|lawsuit|legal action|never again)\b",
    r"\b(worst|terrible|disgusting|horrible)\s+(support|service|experience|course|platform)\b",
    r"\bcancel\s+(my\s+)?(course|subscription|enrollment|account)\b",
]

class SentimentAnalyzer:
    """
    3-class (Positive, Neutral, Negative) sentiment analyzer for customer service.
    Uses NLTK VADER or HuggingFace Emotion/Sentiment transformer pipeline with genuine neutral class.
    Calculates dynamic model-driven confidence scores and handles negation properly.
    """
    def __init__(self, prefer_transformers=False):
        self.engine = None
        self.pipeline = None
        self.vader = None

        if prefer_transformers:
            try:
                from transformers import pipeline
                self.pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None
                )
                self.engine = "transformers"
            except Exception:
                self.engine = None

        if self.engine is None:
            import nltk
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
            self.engine = "vader"

    def _check_escalation_intent(self, text):
        """
        Detects urgent complaints using regex word boundaries.
        Prevents false triggers like 'sue' in 'issue/tissue' or 'refund' in 'What is your refund policy?'.
        """
        return any(re.search(pat, text, re.IGNORECASE) for pat in ESCALATION_PATTERNS)

    def analyze(self, text):
        text = (text or "").strip()
        if not text:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "escalate": False,
                "engine": self.engine,
            }

        is_escalation = self._check_escalation_intent(text)

        if self.engine == "transformers" and self.pipeline is not None:
            preds = self.pipeline(text[:512])[0]
            preds_dict = {p["label"]: float(p["score"]) for p in preds}

            pos_score = preds_dict.get("joy", 0.0)
            neg_score = (
                preds_dict.get("anger", 0.0)
                + preds_dict.get("sadness", 0.0)
                + preds_dict.get("disgust", 0.0)
                + preds_dict.get("fear", 0.0)
            )
            neu_score = preds_dict.get("neutral", 0.0) + preds_dict.get("surprise", 0.0)

            if pos_score > max(neg_score, neu_score) and pos_score >= 0.40:
                sentiment = "positive"
                confidence = round(pos_score, 3)
            elif neg_score > max(pos_score, neu_score) and neg_score >= 0.35:
                sentiment = "negative"
                confidence = round(neg_score, 3)
            else:
                sentiment = "neutral"
                confidence = round(neu_score, 3)
        else:
            scores = self.vader.polarity_scores(text)
            compound = float(scores["compound"])

            # Informational/technical questions with mild positive words (e.g. 'support Mac OS') should be neutral
            is_question = bool(re.search(r"^(do|can|is|what|how|where|why|will|does)\b|\?$", text.strip(), re.IGNORECASE))
            has_strong_sentiment = any(w in text.lower() for w in ["love", "great", "best", "awesome", "terrible", "worst", "hate", "horrible"])

            if is_question and not has_strong_sentiment and abs(compound) < 0.45:
                sentiment = "neutral"
                confidence = round(float(scores["neu"]), 3)
            elif compound >= 0.10:
                sentiment = "positive"
                confidence = round(max(compound, float(scores["pos"])), 3)
            elif compound <= -0.10:
                sentiment = "negative"
                confidence = round(max(abs(compound), float(scores["neg"])), 3)
            else:
                sentiment = "neutral"
                confidence = round(float(scores["neu"]), 3)

        # Escalation policy:
        # Trigger escalate if explicit escalation intent detected, or severe negative sentiment
        escalate = bool(is_escalation or (sentiment == "negative" and confidence >= 0.85))

        if is_escalation:
            sentiment = "negative"
            confidence = round(max(confidence, 0.85), 3)

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "escalate": escalate,
            "engine": self.engine,
        }
