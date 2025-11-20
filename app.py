import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -----------------------------
# Load Vector DB
# -----------------------------
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        collection_name="company_rag_db",
        embedding_function=embeddings,
        persist_directory="vector_store"
    )
    return vectordb

# -----------------------------
# Load Local LLM (FLAN-T5 Base)
# -----------------------------
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# -----------------------------
# Generate Answer
# -----------------------------
def generate_answer(tokenizer, model, retriever, query):
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are an assistant that must answer ONLY using the context provided.
If the answer is not in the context, respond with:
"I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION:
{query}

Provide a detailed answer (3-5 lines):
"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🤖 Local RAG Chatbot")
st.write("Ask anything based on your documents!")

query = st.text_input("🔍 Ask your question:")

if query:
    vectordb = load_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    tokenizer, model = load_llm()

    answer = generate_answer(tokenizer, model, retriever, query)

    st.subheader("🧠 Answer:")
    st.write(answer)
