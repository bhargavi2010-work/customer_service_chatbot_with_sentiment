# langchain_helper.py - RAG pipeline with sentiment analysis

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
_sentiment_analyzer = SentimentAnalyzer()


# create vector database from CSV dataset
def create_vector_db():
    csv_path = "dataset/dataset.csv" if os.path.exists("dataset/dataset.csv") else "dataset.csv"
    loader = CSVLoader(file_path=csv_path, source_column="prompt")
    data = loader.load()

    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)


# build prompt template with sentiment instruction
def _build_prompt(sentiment):
    tone_instruction = build_prompt_instruction(sentiment)

    prompt_template = f"""You are a helpful customer service assistant.
    Use the following context to answer the question if it contains relevant information.
    If the answer is not found in the context, answer the user's question directly, accurately, and politely using your general knowledge.

    TONE INSTRUCTION: {tone_instruction}

    CONTEXT: {{context}}

    QUESTION: {{question}}"""

    return PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )


# create qa chain
def get_qa_chain(sentiment="neutral"):
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()
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

    # fallback to LLM direct generation if chain result is empty or I don't know
    if not raw_answer or "don't know" in raw_answer.lower():
        tone_instruction = build_prompt_instruction(sentiment_result["sentiment"])
        direct_prompt = f"Answer this customer question accurately and politely. TONE: {tone_instruction}\nQuestion: {question}"
        raw_answer = llm.invoke(direct_prompt)

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
