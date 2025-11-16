import os
import torch
import os
import torch
from functools import lru_cache

@lru_cache(maxsize=1)
def build_embedding_model():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    model_name = "NeuML/pubmedbert-base-embeddings"

    # Lazy-loading, cached, device-aware embedding model
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

