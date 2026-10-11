# 🎓 NullClass AI Q&A Chatbot

An intelligent, sentiment-aware student support chatbot built using **LangChain**, **Google Generative AI (Gemini / PaLM)**, **FAISS Vector Database**, and **Streamlit**.

---

## 📌 About the Project

**NullClass** is an e-learning platform that offers data-related courses and virtual internships. With thousands of enrolled students asking repetitive questions on Discord and email, human support staff spend hours answering common queries every day.

This project solves that problem by building an automated **AI-powered Q&A Assistant**:
- Uses the **real FAQ dataset** from NullClass.
- Helps students get instant, accurate answers 24/7 in just seconds.
- Detects the student's emotional tone (happy, neutral, or frustrated) to respond with appropriate empathy.
- Flags urgent issues (like refunds) for human staff assistance.

---

## ✨ Key Features

- **⚡ Instant & Accurate Answers**: Retrieves answers directly from NullClass's official course & internship FAQs.
- **🧠 Semantic Search with FAISS**: Understands user intent even if students ask questions using different words or phrasing.
- **🤖 Powered by Google LLM & LangChain**: Generates natural, complete, and polite answers using RAG (Retrieval-Augmented Generation).
- **🎭 Sentiment Analysis & Tone Adaptation**:
  - 😊 **Positive**: Responds enthusiastically and supportively.
  - 😐 **Neutral**: Gives direct, clear information.
  - 😡 **Negative / Angry**: Expresses empathy, apologizes for inconvenience, and calms the student.
- **🚨 Human Escalation Trigger**: Automatically flags urgent messages (e.g., refund requests, severe complaints) so human mentors can intervene.
- **📊 Live Interactive UI**: Built with Streamlit, showing live sentiment tags, confidence metrics, and a sentiment history trend chart.

---

## 🔄 How It Works (Project Workflow)

```text
                  [ Student Types Question ]
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
  1. Sentiment Analyzer                 2. FAISS Vector Search
  • Detects Positive / Neutral / Angry   • HuggingFace Embeddings
  • Checks for Escalation keywords      • Finds matching FAQ context
            │                                   │
            └─────────────────┬─────────────────┘
                              ▼
                   3. LangChain Prompt Engine
                   • Injects FAQ context
                   • Applies tone instruction based on mood
                              │
                              ▼
                  4. Google Generative AI (LLM)
                  • Generates helpful, polite answer
                              │
                              ▼
                   5. Streamlit Web Interface
                   • Shows answer + sentiment badge + tone adapter
                   • Displays human alert if escalation is needed
```

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM & Framework**: Google Generative AI (`gemini-3.6-flash` / PaLM), LangChain
- **Vector Store & Embeddings**: FAISS, HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Sentiment Detection**: HuggingFace Transformers (`DistilBERT`) with automatic fallback to VADER (`nltk`)
- **Language**: Python 3.10+

---

## 📂 Project Structure

```text
customer_service_chatbot/
│
├── dataset/
│   └── dataset.csv          # Real FAQ dataset from NullClass (Questions & Answers)
├── faiss_index/             # Saved vector database for fast similarity search
│   ├── index.faiss
│   └── index.pkl
├── src/
│   ├── main.py              # Streamlit web app interface
│   ├── langchain_helper.py  # RAG pipeline & LLM response logic
│   ├── sentiment.py         # Sentiment analysis engine
│   ├── response_adapter.py  # Tone formatting & empathetic opener/closers
│   └── test_sentiment.py    # Test script to test sentiment without LLM
├── requirements.txt         # Required Python packages
└── .env                     # Stores GOOGLE_API_KEY (kept private)
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/bhargavi2010-work/customer_service_chatbot_with_sentiment.git
cd customer_service_chatbot_with_sentiment
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Your Google API Key
Create a `.env` file in the root folder and add your key:
```env
GOOGLE_API_KEY="your_google_api_key_here"
```
*(You can get a free key from Google AI Studio / MakerSuite).*

### 4. Run the Streamlit Application
```bash
streamlit run src/main.py
```

### 5. Using the App
1. When you first open the app, click **"Create Knowledgebase"** to build the FAISS index from the FAQ dataset.
2. Once created, type your question in the text box and get instant answers!

---

## 💡 Sample Questions to Try

- *Do you provide internships and also offer EMI payments?*
- *Do you have a JavaScript course?*
- *Should I learn Power BI or Tableau?*
- *I have a Mac computer. Can I use Power BI on it?*
- *I am facing an issue with my assignment and I am really frustrated.* (Tests negative sentiment & tone adaptation)
- *I want a refund immediately!* (Tests human escalation trigger)

---

## 👥 Use Case Summary

Built as an academic GenAI project to demonstrate how **Retrieval-Augmented Generation (RAG)** combined with **Sentiment Analysis** can automate repetitive student support in real-world EdTech companies like NullClass.