# sentiment.py - simple sentiment analyzer for customer support

class SentimentAnalyzer:
    def __init__(self, prefer_transformers=True):
        self.engine = None
        self.pipeline = None
        self.vader = None

        # try loading transformer model
        if prefer_transformers:
            try:
                from transformers import pipeline
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                self.engine = "transformers"
            except Exception:
                self.engine = None

        # fallback to vader if transformers is not available
        if self.engine is None:
            import nltk
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
            self.engine = "vader"

    # check for urgent complaint keywords
    def _has_escalation_keyword(self, text):
        keywords = [
            "unacceptable", "furious", "scam", "refund", "lawsuit", "sue",
            "worst", "terrible", "disgusting", "cancel my", "never again"
        ]
        lowered = text.lower()
        return any(kw in lowered for kw in keywords)

    # check if query is a positive customer intent / course application request
    def _is_positive_intent(self, text):
        lowered = text.lower().strip()
        positive_keywords = [
            "want to apply", "apply for", "interested in", "want to join",
            "eager to", "excited", "looking forward", "love this", "great course",
            "best course", "super helpful", "highly recommend", "enroll in"
        ]
        return any(pk in lowered for pk in positive_keywords)

    # check if query is a standard informational FAQ question
    def _is_faq_question(self, text):
        lowered = text.lower().strip()
        question_words = [
            "can i", "can you", "what is", "what are", "how can", "how do", "how to", "is there",
            "do you", "does this", "why should", "will this", "i have never",
            "where to", "where can", "details about", "tell me about"
        ]
        return any(qw in lowered for qw in question_words)

    # check if user message is an explicit neutral feedback/statement
    def _is_neutral_phrase(self, text):
        lowered = text.lower().strip()
        neutral_keywords = [
            "neither good nor bad", "neither bad nor good", "not good not bad",
            "normal", "average", "okay", "ok", "fine", "so-so", "so so", "moderate", "just fine"
        ]
        return any(nk in lowered for nk in neutral_keywords)

    def analyze(self, text):
        text = (text or "").strip()
        if not text:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "escalate": False,
                "engine": self.engine
            }

        keyword_hit = self._has_escalation_keyword(text)
        is_pos_intent = self._is_positive_intent(text)
        is_neutral_stmt = self._is_neutral_phrase(text)
        is_question = self._is_faq_question(text)

        # positive application / interest override
        if is_pos_intent and not keyword_hit:
            return {
                "sentiment": "positive",
                "confidence": 0.95,
                "escalate": False,
                "engine": self.engine
            }

        # explicit neutral statement override (e.g. "neither good nor bad", "normal", "average")
        if is_neutral_stmt and not keyword_hit:
            return {
                "sentiment": "neutral",
                "confidence": 0.95,
                "escalate": False,
                "engine": self.engine
            }

        # sentiment using transformers
        if self.engine == "transformers":
            raw = self.pipeline(text[:512])[0]
            label = raw["label"].upper()
            score = float(raw["score"])
            
            if score < 0.60 or (is_question and not keyword_hit):
                sentiment = "neutral"
            else:
                sentiment = "positive" if label == "POSITIVE" else "negative"
            confidence = score

        # sentiment using vader fallback
        else:
            scores = self.vader.polarity_scores(text)
            compound = scores["compound"]
            confidence = abs(compound)
            if is_question and not keyword_hit:
                sentiment = "neutral"
            elif compound >= 0.35:
                sentiment = "positive"
            elif compound <= -0.35:
                sentiment = "negative"
            else:
                sentiment = "neutral"

        # keyword override for frustrated queries
        if keyword_hit:
            sentiment = "negative"
            confidence = max(confidence, 0.85)

        # set escalation flag only for genuine complaints/keywords
        escalate = sentiment == "negative" and (confidence >= 0.85 or keyword_hit)

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "escalate": escalate,
            "engine": self.engine
        }
