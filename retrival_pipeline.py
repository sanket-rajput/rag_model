# retrival_pipeline.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------
# LOAD VECTOR STORE
# ---------------------------------------------------------
def load_vector_store(path="vector_store"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )

    return vectordb


# ---------------------------------------------------------
# LOAD LIGHTWEIGHT LOCAL MODEL
# ---------------------------------------------------------
def load_local_llm():
    model_name = "google/flan-t5-base"  # only ~250MB

    print("📌 Loading FLAN-T5-BASE (lightweight)…")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    def llm(prompt: str):
        if not isinstance(prompt, str):
            prompt = str(prompt)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output_tokens = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.3
        )

        return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return llm


# ---------------------------------------------------------
# BUILD RAG CHAIN
# ---------------------------------------------------------
def build_rag_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    template = """
Use ONLY the following context to answer the question.
If the answer is not in the context, say:
"I don’t know based on the documents."

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    llm = load_local_llm()

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    vectordb = load_vector_store()
    rag_chain = build_rag_chain(vectordb)

    print("\n🔥 RAG READY (FLAN-T5-Base + MiniLM + Chroma) 🔥\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        answer = rag_chain.invoke(query)
        print("\nAssistant:", answer, "\n")


if __name__ == "__main__":
    main()
