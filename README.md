MedQuery
PDF-Based Medical Q&A Assistant

MedQuery is a Dockerized, privacy-first, session-isolated Retrieval Augmented Generation (RAG) application that allows users to upload medical PDF documents and ask detailed questions about their content.

The system extracts text from PDFs, chunks them into embeddings, indexes them in FAISS, retrieves the most relevant sections, and produces an accurate medical answer using an LLM — all while keeping each user’s data completely isolated.

Features
🔹 1. PDF Upload & Parsing

Upload one or multiple PDFs

Extracts text using pdfplumber

Deletes the PDF immediately after reading (privacy)

🔹 2. Advanced Chunking & Embeddings

Chunked using LangChain text splitters

Uses PubMedBERT embeddings (NeuML/pubmedbert-base-embeddings)

Embeddings normalized for cosine similarity

Lazy-loaded model (loads only once per session)

🔹 3. FAISS Vectorstore (Per User)

Each user gets their own FAISS index

No cross-user leakage (AC3 compliant)

Stored in server memory only (AC4 compliant)

🔹 4. Retrieval-Augmented Generation (RAG)

MMR (Maximal Marginal Relevance) retrieval

Extracts most relevant chunks

Generates answer + trusted source snippet

Clean metadata (file, page, snippet) returned to the UI

🔹 5. Web UI + REST API

Simple HTML UI served via Flask

Upload PDFs, ask questions, view sources

Health and admin endpoints included

🔹 6. Full Dockerization

Completely isolated environment

Reproducible build

Runs anywhere: Windows, Mac, Linux, EC2, Render, Railway

Gunicorn WSGI server

🔹 7. Session Management & Stats (AC5)

Tracks unique sessions

Tracks uploads, queries, chunks generated

Cleans session on logout


Architecture Overview (AC2–AC5 Compliance)

###  AC2 — Correct RAG Behavior
- Extract → Chunk → Embed → Index → Retrieve → Answer  
- Metadata extraction  
- Source transparency

###  AC3 — Session Isolation
- Per-user FAISS index  
- No cross-user leakage  
- Cookie-based session tracking

###  AC4 — Privacy
- PDFs deleted immediately  
- No persistent storage  
- Logout clears vectorstore + session

###  AC5 — Usage Stats
Tracked server-wide:

- Unique sessions  
- Upload count  
- Query count  
- Chunk count  

---

#  Project Structure

MedQuery/
│
├── app.py # Flask entrypoint
├── Dockerfile # Docker build
├── requirements.txt # Python dependencies
│
── notebook/
│ └── document.ipynb
├── templates/
│ └── index.html # Frontend UI
│
├── services/
│ ├── chunker.py # Chunking logic
│ ├── embedder.py # Lazy-loaded embeddings
│ ├── document_loader.py # PDF parsing operations
│ ├── vectorstore.py # Per-session FAISS vectorstore
│ └── llm.py # RAG answer generation
│
└── uploads/ # Temp folder (emptied each upload)

---

#  Docker Installation (Recommended)

### 1️ Build the image

```bash
docker build -t medquery .

```


### 2 Run the container

```bash
docker run -p 8000:8000 medquery
```

### 3 open the application

```bash
http://localhost:8000
```

### 4 health check

```bash
http://localhost:8000/health
```

# Environment Variables

### create a .env file
```bash
FLASK_ENV=production
SECRET_KEY=your-secret-key
```

# API Endpoints

GET /

Serve frontend UI.

POST /upload

Upload PDFs → extract → chunk → embed → index.

POST /ask

Ask question about uploaded documents.

POST /logout

Clear session + delete user vectorstore.

GET /health

Basic health check.

GET /admin/health

Analytics dashboard (server-level stats).

# Lazy-Loading Explained

The embedding model loads only once, on first use.
After that it stays cached in memory.

Benefits:

Faster startup

Avoids Gunicorn timeouts

Saves memory

Ideal for heavier models (PubMedBERT)


