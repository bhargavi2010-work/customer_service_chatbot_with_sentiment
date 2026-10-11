# Customer Service Chatbot Q&A

An intelligent question-and-answer support system built using LangChain, Google Generative AI (Gemini / PaLM), FAISS Vector Database, and Streamlit for an e-learning platform.

---

## About the Project

NullClass is an e-learning company that offers data-related courses and virtual internships. With thousands of enrolled learners asking questions daily on Discord and email, human support staff spend significant time answering repetitive queries.

This project provides an automated, AI-powered customer service Q&A chatbot:
- Built using a real FAQ dataset used by NullClass support staff.
- Enables students to ask course and internship questions directly and get answers in seconds.
- Analyzes customer sentiment (positive, neutral, or negative) to adapt response tone with empathy.
- Flags urgent inquiries (such as refund requests) for human staff escalation.

---

## Key Features

- Instant FAQ Answers: Retrieves verified answers directly from course and internship FAQ data.
- Semantic Search with FAISS: Understands learner intent using HuggingFace sentence embeddings, even if the question is phrased differently.
- Generative AI Responses: Connects with Google LLM via LangChain to generate clear, polite, and contextual answers.
- Sentiment Analysis & Tone Adaptation:
  - Positive: Responds enthusiastically and helpfully.
  - Neutral: Provides concise, direct answers.
  - Negative: Acknowledges user frustration empathetically before answering.
- Human Escalation Alert: Detects critical complaints (e.g., refund requests or strong dissatisfaction) and flags them for human team follow-up.
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
  - Checks for Escalation keywords      - Finds relevant FAQ context
            |                                   |
            +-----------------+-----------------+
                              |
                              v
                   3. LangChain Prompt Engine
                   - Injects retrieved FAQ context
                   - Applies tone instruction based on user mood
                              |
                              v
                  4. Google Generative AI (LLM)
                  - Generates accurate, tailored response
                              |
                              v
                   5. Streamlit Web Dashboard
                   - Displays answer, sentiment badge, and confidence
                   - Triggers human alert if escalation is flagged
```

---

## Tech Stack

- Frontend: Streamlit
- LLM & Orchestration: Google Generative AI (gemini-3.6-flash / PaLM), LangChain
- Vector Store & Embeddings: FAISS, HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- Sentiment Analysis: HuggingFace Transformers (DistilBERT) with automatic fallback to VADER (nltk)
- Language: Python 3.10+

---

## Project Structure

```text
customer_service_chatbot/
|
|-- dataset/
|   `-- dataset.csv          # FAQ dataset (Questions and Answers)
|-- faiss_index/             # Saved vector database index
|   |-- index.faiss
|   `-- index.pkl
|-- src/
|   |-- main.py              # Main Streamlit web application
|   |-- langchain_helper.py  # RAG pipeline and LLM query logic
|   |-- sentiment.py         # Sentiment analysis module
|   |-- response_adapter.py  # Tone formatting and opener/closer wrapping
|   `-- test_sentiment.py    # Standalone script for sentiment testing
|-- requirements.txt         # Project dependencies
`-- .env                     # Local environment file for GOOGLE_API_KEY
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
- I am facing an issue with my assignment and I am really frustrated.
- I want a refund immediately!

---

## Summary

This project demonstrates how Retrieval-Augmented Generation (RAG) combined with sentiment analysis can automate front-line customer service and student inquiry support in an e-learning organization.