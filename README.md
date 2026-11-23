# Domain-Specific RAG Chatbot for PDF Question Answering

A Streamlit chatbot that answers questions **only** from uploaded PDFs and shows the
source document and page number for every answer.

## Workflow
```
Upload PDFs -> Extract text per page (pypdf) -> Chunk (LangChain splitter)
-> Embed (all-MiniLM-L6-v2) -> Store in FAISS -> User question
-> Retrieve top-4 chunks -> LLM (Groq) with strict prompt -> Answer + source/page
```

## Tech stack
Python, pypdf, LangChain, Sentence Transformers, FAISS, Groq LLM, Streamlit, python-dotenv.

## Setup
```bash
git clone <your-repo-url>
cd domain_rag_chatbot
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env           # then add your GROQ_API_KEY
streamlit run app.py
```
Get a free API key at https://console.groq.com

## Usage
1. Upload one or more PDFs in the sidebar.
2. Click **Process Documents**.
3. Ask questions in the chat box.
4. Expand **Sources** under an answer to see the document, page and snippet.
5. Use **Clear Chat** / **Clear Docs** to reset.

## Features
- Multi-PDF upload with type and size validation (10 MB limit)
- Page-level metadata, empty pages skipped
- Chunk size 800, overlap 120
- FAISS index saved locally and reused
- Strict guardrail prompt with refusal when the answer is missing
- Prompt-injection defence and a "verify high-stakes info" warning

## Limitations
- Scanned PDFs need OCR (not included)
- Answers can still be wrong; always verify important information

## Project structure
```
app.py  rag_pipeline.py  document_loader.py  vector_store.py  prompt.py
requirements.txt  README.md  .env.example  documents/  vector_store/  tests/
```

## Testing
See `tests/test_questions.csv` (15 questions: correct, paraphrased, unavailable, injection).
