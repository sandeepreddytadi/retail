from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from typing import List

BASE = Path("faiss_indexes_online")
EMBEDDINGS = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
INDEX_MAP = {
    "faq": BASE / "faqs_faiss",
    "product": BASE / "products_faiss",
    "policy": BASE / "policies_faiss",
    "store": BASE / "stores_faiss"
}

_loaded = {}

def _load_index(intent: str):
    path = INDEX_MAP.get(intent, INDEX_MAP["faq"])
    path_str = str(path)
    if path_str in _loaded:
        return _loaded[path_str]
    db = FAISS.load_local(path_str, EMBEDDINGS, allow_dangerous_deserialization=True)
    _loaded[path_str] = db
    return db

def retrieve(intent: str, query: str, k: int = 3) -> List[str]:
    db = _load_index(intent)
    docs = db.similarity_search(query, k=k)
    return [d.page_content for d in docs]
