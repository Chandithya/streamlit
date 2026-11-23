"""Module 5 & 6: retrieval + grounded answer generation."""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from prompt import PROMPT, FALLBACK_MESSAGE

load_dotenv()
TOP_K = 4  # suggested range: 3-5


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Add it to your .env file (or Streamlit secrets)."
        )
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
        api_key=api_key,
    )


def retrieve(store, question, k=TOP_K):
    return store.similarity_search_with_score(question, k=k)


def format_context(results):
    parts = []
    for doc, _ in results:
        src, page = doc.metadata.get("source"), doc.metadata.get("page")
        parts.append(f"[Source: {src}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(store, question, k=TOP_K):
    """Returns (answer, sources). sources is a list of dicts."""
    results = retrieve(store, question, k)
    context = format_context(results)
    chain = PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question}).strip()

    sources = []
    if FALLBACK_MESSAGE not in answer:
        seen = set()
        for doc, score in results:
            key = (doc.metadata["source"], doc.metadata["page"])
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "source": doc.metadata["source"],
                    "page": doc.metadata["page"],
                    "snippet": doc.page_content[:300] + "...",
                    "score": float(score),
                }
            )
    return answer, sources
