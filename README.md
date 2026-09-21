
# Generatve AI Q&A: Question and Answer System Based on Google Palm LLM and Langchain for E-learning company  

This is an end to end LLM project based on Google Palm and Langchain. We are building a Q&A system for an e-learning company called Nullclass.Nullclass sells data related courses and virtual internships. They have thousands of learners who uses discord server or email to ask questions. This system will provide a streamlit based user interface for students where they can ask questions and get answers. 


## Project Highlights

- Use a real CSV file of FAQs that Nullclass company is using right now. 
- Their human staff will use this file to assist their course learners.
- We will build an LLM based question and answer system that can reduce the workload of their human staff.
- Students should be able to use this system to ask questions directly and get answers within seconds



## Installation

1.Clone this repository to your local machine using:

```bash
  git clone https://github.com/aslin72/customer_service_chatbot_LLM.git
```
2.Navigate to the project directory:

```bash
  cd customer_service_chatbot_LLM
```
3. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
4.Acquire an api key through makersuite.google.com and put it in .env file

```bash
  GOOGLE_API_KEY="your_api_key_here"
```
## Usage

1. Run the Streamlit app by executing:
```bash
streamlit run main.py

```

2.The web app will open in your browser.

- To create a knowledebase of FAQs, click on Create Knolwedge Base button. It will take some time before knowledgebase is created so please wait.

- Once knowledge base is created you will see a directory called faiss_index in your current folder

- Now you are ready to ask questions. Type your question in Question box and hit Enter

## Sample Questions
  - Do you guys provide internship and also do you offer EMI payments?
  - Do you have javascript course?
  - Should I learn power bi or tableau?
  - I've a MAC computer. Can I use powerbi on it?
  - I don't see power pivot. how can I enable it?

## Sentiment Analysis

Every incoming question is passed through a sentiment analyzer before the
answer is generated:

- **Detection**: uses a HuggingFace `transformers` sentiment model
  (`distilbert-base-uncased-finetuned-sst-2-english`), classified into
  `positive` / `neutral` / `negative`. If `transformers`/`torch` aren't
  installed, it automatically falls back to VADER (`nltk`) so the app still
  runs.
- **Response adaptation**:
  - The LLM's own prompt is given a tone instruction matching the detected
    sentiment (e.g. acknowledge frustration before answering, for negative
    messages).
  - The final reply is wrapped with a short, deterministic opener/closer
    per sentiment (e.g. an empathetic opener for negative messages, a
    friendly closer for positive ones), so the tone is consistent even if
    the LLM doesn't fully follow the prompt instruction.
  - Strongly negative messages (high-confidence negative, or containing
    phrases like "refund", "unacceptable", "cancel my") are flagged with
    `escalate: True` and shown a notice that a human agent will follow up.
- **UI**: the Streamlit app shows the detected sentiment badge and
  confidence under each answer, and a warning banner when a message is
  flagged for escalation.

Files:
- `sentiment.py` — sentiment detection (`SentimentAnalyzer`)
- `response_adapter.py` — tone instructions + response wrapping
- `test_sentiment.py` — quick standalone check against sample messages
  (`python test_sentiment.py`), no API key or vector DB needed

## Project Structure

- `src/main.py`: The main Streamlit application script.
- `src/langchain_helper.py`: RAG pipeline and LLM query helper.
- `src/sentiment.py`: Sentiment detection engine.
- `src/response_adapter.py`: Response adaptation layer.
- `src/test_sentiment.py`: Automated sentiment test suite.
- `requirements.txt`: Python package dependencies.
- `.env` *(create locally on your machine)*: Local configuration file for storing your `GOOGLE_API_KEY`.