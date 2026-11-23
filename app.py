"""Module 7: Streamlit interface."""
import streamlit as st
from document_loader import validate_file, load_pdf, MAX_FILE_MB
from vector_store import build_vector_store, save_vector_store, load_vector_store
from rag_pipeline import answer_question

st.set_page_config(page_title="Domain RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 Domain-Specific RAG Chatbot")
st.caption("Ask questions about your uploaded PDFs. Answers come only from your documents.")

# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = load_vector_store()  # reuse saved index if present
if "files" not in st.session_state:
    st.session_state.files = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ---------- sidebar ----------
with st.sidebar:
    st.header("1. Upload documents")
    uploads = st.file_uploader(
        f"PDF files (max {MAX_FILE_MB} MB each)",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploads:
        st.write("**Selected files:**")
        for f in uploads:
            st.write(f"• {f.name}")

    if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
        if not uploads:
            st.warning("Please upload at least one PDF.")
        else:
            all_docs, names = [], []
            for f in uploads:
                ok, msg = validate_file(f)
                if not ok:
                    st.error(msg)
                    continue
                try:
                    all_docs.extend(load_pdf(f, f.name))
                    names.append(f.name)
                except ValueError as e:
                    st.error(str(e))
            if all_docs:
                with st.spinner("Chunking and creating embeddings..."):
                    try:
                        store, n_chunks = build_vector_store(all_docs)
                        save_vector_store(store)
                        st.session_state.store = store
                        st.session_state.files = names
                        st.session_state.messages = []
                        st.success(f"Processed {len(names)} file(s) into {n_chunks} chunks.")
                    except ValueError as e:
                        st.error(str(e))
            elif uploads:
                st.error("No readable text found (scanned PDFs need OCR).")

    if st.session_state.files:
        st.info("Active documents:\n\n" + "\n".join(f"- {n}" for n in st.session_state.files))

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if c2.button("♻️ Clear Docs", use_container_width=True):
        st.session_state.store = None
        st.session_state.files = []
        st.session_state.messages = []
        st.session_state.uploader_key += 1
        st.rerun()

    st.divider()
    st.caption("⚠️ AI answers can be wrong. Please verify high-stakes information in the original document.")

# ---------- chat ----------
def show_sources(sources):
    if sources:
        with st.expander("📚 Sources"):
            for s in sources:
                st.markdown(f"**{s['source']}** — page {s['page']}")
                st.caption(s["snippet"])

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_sources(m.get("sources"))

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if st.session_state.store is None:
            answer, sources = "Please upload and process at least one PDF first.", []
            st.markdown(answer)
        else:
            with st.spinner("Searching documents..."):
                try:
                    answer, sources = answer_question(st.session_state.store, question)
                except Exception as e:
                    answer, sources = f"Error: {e}", []
            st.markdown(answer)
            show_sources(sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
