# vectorstore.py

import os
import re
import pickle
from typing import List, Optional

import numpy as np
import faiss
from langchain_core.documents import Document

from .embedder import build_embedding_model  # same model as chunker


class EmbeddingVectorStore:

    def __init__(
        self,
        embedding_model=None,
        index_path: str = "db/faiss_index.bin",
        metadata_path: str = "db/faiss_metadata.pkl",
    ):
        # Reuse same model as everywhere else
        self.embedding_model = embedding_model or build_embedding_model()

        self.texts: List[str] = []
        self.metadatas: List[dict] = []
        self.index: Optional[faiss.IndexFlatIP] = None

        self.index_path = index_path
        self.metadata_path = metadata_path
        self.doc_embeddings = None

        # Make sure folder exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

    

    def _clean_text(self, text: str) -> str:
        
        text = re.sub(r"<pad>|<EOS>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    

    def embed_docs(self, documents):
      """
      documents: list[Document]
      returns: np.ndarray of shape (n_docs, dim)
      """

      # Extract text
      extracted_text = [self._clean_text(doc.page_content) for doc in documents]

      # Save original text + metadata
      self.texts = extracted_text
      self.metadatas = [doc.metadata for doc in documents]

      # --- Embed documents ---
      embeddings = self.embedding_model.embed_documents(extracted_text)
      embeddings = np.array(embeddings, dtype="float32")

      # Normalize once
      faiss.normalize_L2(embeddings)

      # --- Save for retrieval ---
      self.doc_embeddings = embeddings        # used by retrieve_mmr()
      self.embeddings = embeddings            # optional, but good to keep

      return embeddings


    def build_faiss_index(self, embeddings: np.ndarray) -> None:
        """
        Build a FAISS index from embeddings and save index + metadata to disk.
        """
        embeddings = np.array(embeddings, dtype="float32")
        dim = embeddings.shape[1]

        # Cosine similarity via inner product (since embeddings are normalized)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Save index
        faiss.write_index(index, self.index_path)
        self.index = index

        # Save texts + metadata for later retrieval
        with open(self.metadata_path, "wb") as f:
            pickle.dump(
                {
                    "texts": self.texts,
                    "metadatas": self.metadatas,
                },
                f,
            )

    

    def load_faiss(self) -> None:
        """
        Load FAISS index + metadata from disk.
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found at: {self.index_path}")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {self.metadata_path}")

        self.index = faiss.read_index(self.index_path)

        with open(self.metadata_path, "rb") as f:
            meta = pickle.load(f)
            self.texts = meta.get("texts", [])
            self.metadatas = meta.get("metadatas", [])

    

    def _mmr(self, query_emb, doc_embs, k=5, lambda_mult=0.5):
        
        # similarity to query
        sim_to_query = np.dot(doc_embs, query_emb)

        # similarity between docs
        sim_between_docs = np.dot(doc_embs, doc_embs.T)

        selected = []
        candidates = list(range(len(doc_embs)))

        # choose first = best to query
        first = int(np.argmax(sim_to_query))
        selected.append(first)
        candidates.remove(first)

        while len(selected) < k and candidates:
            mmr_score = []
            for c in candidates:
                diversity = max(sim_between_docs[c][s] for s in selected)
                score = lambda_mult * sim_to_query[c] - (1 - lambda_mult) * diversity
                mmr_score.append((score, c))

            _, best_c = max(mmr_score, key=lambda x: x[0])
            selected.append(best_c)
            candidates.remove(best_c)

        return selected

    

    def retrieve_mmr(self, query: str, top_k: int = 5, pool_size: int = 20):
       
        if self.index is None:
            raise ValueError("FAISS index is not initialized. Call load_faiss() or build_faiss_index() first.")

        # Embed query
        q = np.array(self.embedding_model.embed_query(query), dtype="float32")
        q = q.reshape(1, -1)
        faiss.normalize_L2(q)

        # Get a larger candidate pool from FAISS
        scores, idxs = self.index.search(q, pool_size)
        idxs = idxs[0]
        scores = scores[0]

        # Filter out invalid indices
        valid_mask = idxs >= 0
        idxs = idxs[valid_mask]
        scores = scores[valid_mask]

        if len(idxs) == 0:
            return []

        
        doc_embs = self.doc_embeddings[idxs]

        

        # Run MMR on the candidate pool
        selected_idx_pos = self._mmr(q.flatten(), doc_embs, k=top_k, lambda_mult=0.5)

        results = []
        for rank, pos in enumerate(selected_idx_pos, start=1):
            real_idx = int(idxs[pos])
            results.append(
                {
                    "rank": rank,
                    "text": self.texts[real_idx],
                    "metadata": self.metadatas[real_idx],
                    "score": float(scores[pos]),
                }
            )

        return results
