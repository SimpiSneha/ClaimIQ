"""
Builds the vector index from policy documents.
Run this once (and again any time you add/change policy documents):
    python build_index.py

This reads all .txt files from data/policy_docs/, chunks them,
embeds them with Gemini, and saves a FAISS index to disk so the
RAG agent doesn't have to re-embed everything on every run.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

POLICY_DOCS_DIR = Path(__file__).parent.parent / "data" / "policy_docs"
INDEX_SAVE_PATH = Path(__file__).parent.parent / "data" / "faiss_index"


def build_index():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Copy .env.example to .env and add your key."
        )

    # 1. Load all policy documents
    all_docs = []
    txt_files = list(POLICY_DOCS_DIR.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No .txt files found in {POLICY_DOCS_DIR}")

    print(f"Found {len(txt_files)} policy document(s):")
    for file_path in txt_files:
        print(f"  - {file_path.name}")
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        # Tag each doc with its source filename so we can cite it later
        for doc in docs:
            doc.metadata["source"] = file_path.name
        all_docs.extend(docs)

    # 2. Chunk the documents
    # chunk_size=1000, overlap=150 keeps most clauses intact while giving
    # enough context per chunk for the LLM to answer accurately.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks")

    # 3. Embed and store in FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 4. Save to disk so we don't have to re-embed every time
    vector_store.save_local(str(INDEX_SAVE_PATH))
    print(f"Index saved to {INDEX_SAVE_PATH}")


if __name__ == "__main__":
    build_index()