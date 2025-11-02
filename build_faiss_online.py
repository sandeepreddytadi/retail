# build_faiss_online.py
"""
Build FAISS indexes using FREE local embeddings - NO API LIMITS!
"""
import os
from pathlib import Path
import re
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

DATA_FILE = Path("data") / "finalData.txt"
OUT_DIR = Path("faiss_indexes_online")
OUT_DIR.mkdir(exist_ok=True, parents=True)

def read_file(path):
    return Path(path).read_text(encoding="utf-8")

def split_sections(text):
    sections = {"faqs": "", "products": "", "policies": "", "stores": ""}
    m = re.search(r"Faqs(.*?)(?:Product Catelogue|Product Catalogue|Product Description)", text, re.S | re.I)
    if m: sections["faqs"] = m.group(1).strip()
    m = re.search(r"Product Description(.*?)(?:Return Policy|Return Policy\s|Return Policy)", text, re.S | re.I)
    if m: sections["products"] = m.group(1).strip()
    m = re.search(r"Return Policy(.*?)(?:Store information|Store Information|Store Number)", text, re.S | re.I)
    if m: sections["policies"] = m.group(1).strip()
    m = re.search(r"(Store information|Store Information|Store Number)(.*)", text, re.S | re.I)
    if m: sections["stores"] = m.group(2).strip()
    if not sections["products"]:
        m2 = re.search(r"Product Catelogue(.*?)(?:Return Policy|Store information|Store Information|$)", text, re.S | re.I)
        if m2: sections["products"] = m2.group(1).strip()
    return sections

def chunk_text(text, chunk_size=400, overlap=60):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def build_index_for_section(name, content, embeddings):
    if not content.strip(): 
        print(f"[WARN] No content for {name}; skipping.")
        return
    chunks = chunk_text(content)
    docs = [Document(page_content=c) for c in chunks]
    db = FAISS.from_documents(docs, embeddings)
    out = OUT_DIR / f"{name}_faiss"
    db.save_local(str(out))
    print(f"[OK] Saved {name} index to {out}")

def main():
    assert DATA_FILE.exists(), f"Missing {DATA_FILE}"
    text = read_file(DATA_FILE)
    sections = split_sections(text)
    
    # FREE local embeddings - no rate limits!
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    for k, v in sections.items():
        build_index_for_section(k, v, embeddings)
    print("✅ FAISS indexes built successfully using FREE local embeddings!")

if __name__ == "__main__":
    main()