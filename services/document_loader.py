# document_loader.py
from typing import List
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
import os

def load_pdf(pdf_path: str) -> List[Document]:
    """
    Load a single PDF file and return a list of LangChain Document objects.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    return docs


def load_pdfs(pdf_paths: List[str]) -> List[Document]:
    """
    Load multiple PDF files and return a single combined list of Documents.

    """
    all_docs: List[Document] = []

    for path in pdf_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF file not found: {path}")
        loader = PDFPlumberLoader(path)
        docs = loader.load()
        all_docs.extend(docs)

    return all_docs
