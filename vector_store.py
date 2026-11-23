"""Module 3 & 4: chunking, embeddings and FAISS vector store."""
import os
from functools import lru_cache
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800       # suggested range: 700-1000
CHUNK_OVERLAP = 120    # suggested range: 100-150
INDEX_DIR = os.path.join("vector_store", "saved_index")


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity
    )


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)  # metadata is copied to each chunk


def build_vector_store(docs):
    chunks = split_documents(docs)
    if not chunks:
        raise ValueError("No text could be extracted from the uploaded PDFs.")
    store = FAISS.from_documents(chunks, get_embeddings())
    return store, len(chunks)


def save_vector_store(store, path=INDEX_DIR):
    os.makedirs(path, exist_ok=True)
    store.save_local(path)


def load_vector_store(path=INDEX_DIR):
    if not os.path.exists(os.path.join(path, "index.faiss")):
        return None
    return FAISS.load_local(
        path, get_embeddings(), allow_dangerous_deserialization=True
    )
