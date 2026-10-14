# langchain_helper.py - Grounded RAG pipeline with sentiment analysis and tone adaptation

import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

from sentiment import SentimentAnalyzer
from response_adapter import build_prompt_instruction, shape_response

load_dotenv()

# initialize LLM model and embeddings
api_key = os.environ.get("GOOGLE_API_KEY", "")
llm = GoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key, temperature=0.1)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb_file_path = "faiss_index"
_sentiment_analyzer = SentimentAnalyzer(prefer_transformers=False)


# create vector database from CSV dataset
def create_vector_db():
    csv_path = "dataset/dataset.csv" if os.path.exists("dataset/dataset.csv") else "dataset.csv"
    loader = CSVLoader(file_path=csv_path, source_column="prompt")
    data = loader.load()

    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)


# build prompt template with sentiment instruction and strict grounding
def _build_prompt(sentiment):
    tone_instruction = build_prompt_instruction(sentiment)

    prompt_template = f"""You are an official customer support assistant for NullClass e-learning platform.

CRITICAL INSTRUCTIONS:
1. Answer the question accurately and honestly using ONLY the factual information provided in the FAQ CONTEXT below.
2. If the answer is NOT present in the provided FAQ context, do NOT invent, guess, or fabricate company policies, course pricing, refund terms, or timeline promises. Instead, state clearly:
   "I apologize, but that specific detail is not covered in our official FAQ database. Please reach out to our support team at support@nullclass.com or on our Discord community, and we will be happy to assist you."
3. TONE & EMPATHY INSTRUCTION: {tone_instruction}

FAQ CONTEXT:
{{context}}

CUSTOMER QUESTION:
{{question}}

OFFICIAL RESPONSE:"""

    return PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )


# create qa chain
def get_qa_chain(sentiment="neutral"):
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    PROMPT = _build_prompt(sentiment)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return chain


# main function to get response with sentiment
def get_response(question):
    sentiment_result = _sentiment_analyzer.analyze(question)

    chain = get_qa_chain(sentiment=sentiment_result["sentiment"])
    raw = chain(question)
    raw_answer = raw.get("result", "")
    source_docs = raw.get("source_documents", [])

    # If the retrieved context cannot answer or result is empty, use safe grounded fallback
    if not raw_answer or "don't know" in raw_answer.lower() or "not mentioned" in raw_answer.lower():
        raw_answer = (
            "I apologize, but I do not have verified information regarding your inquiry in our official FAQ database. "
            "Please reach out directly to our support desk at support@nullclass.com or on our official Discord channel so an administrator can assist you."
        )

    # Shape response with sentiment opener, closer, and escalation banner if needed
    final_answer = shape_response(raw_answer, sentiment_result)

    return {
        "answer": final_answer,
        "raw_answer": raw_answer,
        "sentiment": sentiment_result["sentiment"],
        "confidence": sentiment_result["confidence"],
        "escalate": sentiment_result["escalate"],
        "source_documents": source_docs,
    }


if __name__ == "__main__":
    create_vector_db()
    result = get_response("This is ridiculous, nothing works and I want a refund!")
    print(result)
