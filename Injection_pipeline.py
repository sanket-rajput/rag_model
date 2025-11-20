# injection_pipeline.py
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

def load_files(directory_path):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory {directory_path} does not exist.")

    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=lambda path: TextLoader(
            path,
            encoding="utf-8-sig",
            autodetect_encoding=True
        )
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError("No documents found in the specified directory.")

    for i, doc in enumerate(documents):
        print(f"Document {i+1} preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
        print("-" * 40)

    print(f"Loaded {len(documents)} documents.")
    return documents


def chunk_documents(documents, chunk_size=300, chunk_overlap=50):
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (chunk_size=300, overlap=50).")
    return chunks



def create_vector_store(chunks, persist_directory="vector_store"):
    # FREE local embeddings (no OpenAI required)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating vector store (explicit constructor + add_documents)...")

    vectordb = Chroma(
        collection_name="company_rag_db",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    vectordb.add_documents(chunks)

    try:
        vectordb.persist()
    except:
        pass

    print(f"Vector store created and persisted at: {persist_directory}")
    return vectordb




def main():
    print("Main function (ingestion)")
    documents = load_files(directory_path="docs")
    chunks = chunk_documents(documents)
    create_vector_store(chunks, persist_directory="vector_store")


if __name__ == "__main__":
    main()
