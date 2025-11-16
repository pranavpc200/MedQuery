import re
import os
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# preprocessing the text to remove \n
import re
def clean_text(text: str) -> str:
    """
    Cleans up text by removing extra newlines and redundant spaces.
    Keeps paragraph spacing intact.
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Remove newline characters within paragraphs
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Normalize multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    
    # Fix hyphenated line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Remove hyphen + space (e.g., 'pri- marily' → 'primarily')
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    return text.strip()


def split_documents(
    documents: List[Document],
    embedding_model
) -> List[Document]:
    """
    Splits documents into semantic chunks using LangChain's SemanticChunker.

    """

    # Convert all Document pages → clean text
    cleaned_docs = []
    for doc in documents:
        meta = dict(doc.metadata)
        source_path = meta.get("source", "")

        if source_path:
          meta["file_name"] = os.path.basename(source_path)

        text = clean_text(doc.page_content)
        cleaned_docs.append(Document(page_content=text, metadata=meta))


    # Semantic Chunker
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(cleaned_docs)

    # Assign numeric ID per chunk
    for i, doc in enumerate(chunks):
        doc.metadata = doc.metadata or {}
        doc.metadata["chunk_id"] = i

    return chunks