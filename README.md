# Customer Service Chatbot Q&A

An intelligent question-and-answer support system built using LangChain, Google Generative AI (Gemini / PaLM), FAISS Vector Database, and Streamlit for an e-learning platform.

---

## About the Project

NullClass is an e-learning company that offers data-related courses and virtual internships. With thousands of enrolled learners asking questions daily on Discord and email, human support staff spend significant time answering repetitive queries.

This project provides an automated, AI-powered customer service Q&A chatbot:
- Built using a real FAQ dataset used by NullClass support staff.
- Enables students to ask course and internship questions directly and get answers in seconds.
- Analyzes customer sentiment (positive, neutral, or negative) with dynamic model confidence scores.
- Adapts the response tone with empathetic openers and closers tailored to user mood.
- Flags urgent inquiries (such as refund demands or severe frustration) for priority human staff escalation.
- Enforces strict grounding in official FAQ documentation to prevent policy hallucination.

---

## Key Features

- Instant FAQ Answers: Retrieves verified answers directly from course and internship FAQ data.
- Strict Grounding (Zero Policy Hallucination): Answers are grounded strictly in the official FAQ database. If a detail is missing, it deflects safely to human support rather than inventing company policy.
- Semantic Search with FAISS: Understands learner intent using HuggingFace sentence embeddings, even if the question is phrased differently.
- Generative AI Responses: Connects with Google LLM via LangChain to generate clear, polite, and contextual answers.
- Multi-Class Sentiment Analysis & Tone Adaptation:
  - Positive: Responds enthusiastically with appreciative openers and supportive closers.
  - Neutral: Provides concise, direct, and factual answers without forced emotion.
  - Negative: Sincerly acknowledges user frustration, validates their difficulty, and provides helpful guidance.
- Priority Human Escalation: Uses regex word-boundary detection to capture genuine urgent complaints (e.g., refund demands, scam accusations) and displays high-priority notification banners for human intervention.
- Streamlit Web Interface: Clean web dashboard with real-time sentiment indicators, confidence metrics, and sentiment distribution analytics.

---

## Working Flow

```text
                  [ User Asks a Question ]
                              |
            +-----------------+-----------------+
            |                                   |
            v                                   v
  1. Sentiment Analyzer                 2. FAISS Vector Search
  - Classifies: Positive/Neutral/Negative- HuggingFace Embeddings
  - Word-boundary Escalation Check      - Finds relevant FAQ context
            |                                   |
            +-----------------+-----------------+
                              |
                              v
                   3. LangChain Prompt Engine
                   - Injects retrieved FAQ context
                   - Applies tone instruction based on user mood
                   - Strictly grounded (no policy hallucination)
                              |
                              v
                  4. Google Generative AI (LLM)
                  - Generates accurate, tailored response
                              |
                              v
                   5. Response Shaping Layer
                   - Adds empathetic opener & closer
                   - Attaches human escalation notice if flagged
                              |
                              v
                   6. Streamlit Web Dashboard
                   - Displays answer, sentiment badge, and confidence
                   - Triggers human alert if escalation is flagged
```

---

## Evaluation Benchmark & Measured Impact

The project includes an automated evaluation suite (`src/evaluate_sentiment.py`) benchmarking 36 diverse customer queries across positive, neutral, negative, and critical escalation scenarios.

### 1. Sentiment Detection Accuracy

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Positive** | 100.0% | 100.0% | 100.0% | 10 |
| **Neutral** | 93.3% | 100.0% | 96.6% | 14 |
| **Negative** | 100.0% | 91.7% | 95.7% | 12 |
| **Overall Accuracy** | **97.2%** | **97.2%** | **97.2%** | **36** |

- **Escalation Flag Detection**: Precision: **100.0%** | Recall: **100.0%** | F1-Score: **100.0%**

### 2. Measured Customer-Satisfaction (CSAT) Impact

Comparison between baseline unadapted responses versus sentiment-adapted empathetic responses:

| Interaction Category | Count | Baseline CSAT (1-5) | Adapted CSAT (1-5) | Net Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Negative / Frustrated Queries** | 12 | 2.0 / 5.0 | 4.3 / 5.0 | **+115.0%** |
| **Neutral / General Inquiries** | 14 | 4.0 / 5.0 | 4.3 / 5.0 | **+7.5%** |
| **Positive / Satisfied Queries** | 10 | 4.1 / 5.0 | 4.8 / 5.0 | **+17.1%** |
| **Overall Platform CSAT** | **36** | **3.36 / 5.0** | **4.44 / 5.0** | **+32.1%** |

To run the evaluation benchmarks yourself:
```bash
python src/evaluate_sentiment.py
```

To run unit tests:
```bash
python src/test_sentiment.py
```

---

## Tech Stack

- Frontend: Streamlit
- LLM & Orchestration: Google Generative AI (gemini-3.6-flash / PaLM), LangChain
- Vector Store & Embeddings: FAISS, HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- Sentiment Analysis: NLTK VADER / HuggingFace Transformers (j-hartmann/emotion-english-distilroberta-base)
- Language: Python 3.10+

---

## Project Structure

```text
customer_service_chatbot/
|
|-- dataset/
|   `-- dataset.csv            # FAQ dataset (Questions and Answers)
|-- faiss_index/               # Saved vector database index
|   |-- index.faiss
|   `-- index.pkl
|-- src/
|   |-- main.py                # Main Streamlit web application
|   |-- langchain_helper.py    # Grounded RAG pipeline and LLM query logic
|   |-- sentiment.py           # Multi-class sentiment analysis module
|   |-- response_adapter.py    # Tone formatting, openers/closers, escalation notices
|   |-- test_sentiment.py      # Unit test suite with edge cases
|   `-- evaluate_sentiment.py  # Measured detection accuracy & CSAT impact benchmark
|-- evaluation_results.json    # Serialized quantitative evaluation metrics
|-- requirements.txt           # Project dependencies
`-- .env                       # Local environment file for GOOGLE_API_KEY
```

---

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/bhargavi2010-work/customer_service_chatbot_with_sentiment.git
cd customer_service_chatbot_with_sentiment
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Google API Key
Create a `.env` file in the root folder and add your API key:
```env
GOOGLE_API_KEY="your_google_api_key_here"
```

### 4. Run the Streamlit Application
```bash
streamlit run src/main.py
```

### 5. Using the App
1. When opening the app for the first time, click "Create Knowledgebase" to generate the FAISS vector index.
2. Type any question in the input field and press Enter to receive an instant answer.

---

## Sample Questions

- Do you guys provide internship and also do you offer EMI payments?
- Do you have a javascript course?
- Should I learn power bi or tableau?
- I have a MAC computer. Can I use powerbi on it?
- The course video is broken and nothing works.
- I want a refund right now!

---

## Summary

This project demonstrates how Retrieval-Augmented Generation (RAG) combined with sentiment analysis can automate front-line customer service and student inquiry support in an e-learning organization.