import os
import uuid

from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from services.document_loader import load_pdf, load_pdfs
from services.embedder import build_embedding_model
from services.chunker import split_documents
from services.vectorstore import EmbeddingVectorStore
from services.llm import rag_answer

# Load .env
load_dotenv()

app = Flask(__name__)
app.secret_key = "your-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- UI ROUTE ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ---------- HEALTH CHECK ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200



# ---------- SESSION STORE (AC3) ----------
SESSION_STORES = {}

# ---------- USAGE STATS (AC5) ----------
STATS = {
    "total_queries": 0,
    "total_uploads": 0,
    "total_chunks": 0,
    "session_count": 0,
    "tracked_sessions": set()   # <-- FIXED
}


# ---------- SESSION ID UTILITY ----------
def get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]



# ------------------------------------------------------------
# ---------- UPLOAD & BUILD INDEX (AC2 + AC3 + AC4) ----------
# ------------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload_files():

    # (1) Identify session
    session_id = get_session_id()

    # (2) Create a fresh vector store for THIS USER ONLY (AC3)
    vs = EmbeddingVectorStore(embedding_model=build_embedding_model())
    SESSION_STORES[session_id] = vs

    # -------- FILE UPLOAD LOGIC --------
    uploaded_files = request.files.getlist("files")

    if not uploaded_files or uploaded_files == [None]:
        return jsonify({"error": "No files uploaded"}), 400

    saved_paths = []
    for f in uploaded_files:
        if not f or f.filename == "":
            continue
        filename = secure_filename(f.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(save_path)
        saved_paths.append(save_path)

    if not saved_paths:
        return jsonify({"error": "No valid files uploaded"}), 400

    # -------- LOAD PDF CONTENT --------
    docs = load_pdf(saved_paths[0]) if len(saved_paths) == 1 else load_pdfs(saved_paths)

    if not docs:
        return jsonify({"error": "No text could be extracted from PDFs"}), 400

    # -------- AC4: DELETE PDFs IMMEDIATELY --------
    for p in saved_paths:
        try:
            os.remove(p)
        except:
            pass

    # -------- CHUNKING --------
    chunks = split_documents(docs, embedding_model = build_embedding_model)
    if not chunks:
        return jsonify({"error": "Chunking produced no chunks"}), 500

    # -------- AC5 STATS --------
    STATS["total_uploads"] += 1
    STATS["total_chunks"] += len(chunks)

    # Count unique sessions properly
    if session_id not in STATS["tracked_sessions"]:
        STATS["tracked_sessions"].add(session_id)
        STATS["session_count"] += 1

    # -------- EMBEDDINGS + FAISS INDEX (AC3) --------
    embeddings = vs.embed_docs(chunks)
    vs.build_faiss_index(embeddings)

    return jsonify({
        "message": "Files processed and index built successfully",
        "files_uploaded": len(saved_paths),
        "doc_pages_loaded": len(docs),
        "chunks_created": len(chunks),
    }), 200



# ------------------------------------------------------------
# ---------- ASK / Q&A (AC2 + AC3) ----------
# ------------------------------------------------------------
@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json(silent=True) or {}
    question = data.get("question") or data.get("query")

    if not question:
        return jsonify({"error": "Field 'question' (or 'query') is required"}), 400

    # (1) Identify user session
    session_id = get_session_id()

    # AC5 stats
    STATS["total_queries"] += 1

    # (2) Ensure the user has an index (AC3)
    if session_id not in SESSION_STORES:
        return jsonify({
            "error": "No index found for this session. Please upload PDFs first."
        }), 400

    vs = SESSION_STORES[session_id]

    # (3) Ensure FAISS index exists (AC3)
    if vs.index is None or not vs.texts:
        return jsonify({
            "error": "Your session index is empty. Upload PDFs first."
        }), 400

    # Retrieve chunks
    top_k = int(data.get("top_k", 4))
    pool_size = int(data.get("pool_size", 20))

    retrieval_results = vs.retrieve_mmr(
        query=question,
        top_k=top_k,
        pool_size=pool_size,
    )

    # Generate LLM answer
    result = rag_answer(question, retrieval_results)

    # -------- AC2: extract clean metadata --------
    top = result["used_chunks"][0] if result["used_chunks"] else None

    if top:
        meta = top["metadata"] or {}
        file_name = meta.get("file_name") or meta.get("source", "").split("/")[-1]
        page = meta.get("page", "?")
        snippet = top.get("text", "")[:400]

        source_info = {
            "file": file_name,
            "page": page,
            "snippet": snippet
        }
    else:
        source_info = None

    return jsonify({
        "answer": result["answer"],
        "source": source_info
    }), 200



# ------------------------------------------------------------
# ---------- LOGOUT / ERASE SESSION DATA (AC4) ----------
# ------------------------------------------------------------
@app.route("/logout", methods=["POST"])
def logout():
    session_id = session.get("session_id")

    # Delete vectorstore for this user
    if session_id in SESSION_STORES:
        del SESSION_STORES[session_id]

    # Clear session cookie
    session.clear()

    return jsonify({"message": "Session cleared. All user data erased."}), 200



# ------------------------------------------------------------
# ---------- ADMIN HEALTH ENDPOINT (AC5) ----------
# ------------------------------------------------------------
@app.route("/admin/health", methods=["GET"])
def admin_health():
    status = "ok"

    # Example health condition
    if len(SESSION_STORES) > 50:
        status = "high_load"

    return jsonify({
        "status": status,
        "active_sessions": len(SESSION_STORES),
        "total_sessions": STATS["session_count"],
        "total_queries": STATS["total_queries"],
        "total_uploads": STATS["total_uploads"],
        "total_chunks": STATS["total_chunks"],
    }), 200



# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

