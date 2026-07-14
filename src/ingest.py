import os
import json
import tiktoken
import chromadb
import PyPDF2
from google import genai
from config import *

# client = genai.Client(api_key=GEMINI_API_KEY)

# def embed_texts(texts: list[str]) -> list[list[float]]:
#     result = client.models.embed_content(
#         model="models/gemini-embedding-001",
#         contents=texts
#     )
#     return [e.values for e in result.embeddings]
from sentence_transformers import SentenceTransformer
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: list[str]) -> list[list[float]]:
    return _embedder.encode(texts).tolist()
def extract_pdf(path: str) -> str:
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def load_documents(corpus_path: str) -> list[dict]:
    docs = []
    if not os.path.exists(corpus_path):
        raise SystemExit(f"ERROR: corpus/ not found")
    files = [f for f in os.listdir(corpus_path)
             if not f.startswith(".")]
    if not files:
        raise SystemExit("ERROR: corpus/ is empty")
    for fname in files:
        fpath = os.path.join(corpus_path, fname)
        try:
            if fname.endswith(".pdf"):
                text = extract_pdf(fpath)
                docs.append({"text": text,
                             "source": fname,
                             "doc_type": "pdf"})
            elif fname.endswith(".txt"):
                text = open(fpath).read()
                docs.append({"text": text,
                             "source": fname,
                             "doc_type": "txt"})
            elif fname.endswith((".json",".yaml",".yml")):
                text = open(fpath).read()
                docs.append({"text": text,
                             "source": fname,
                             "doc_type": "structured"})
        except Exception as e:
            print(f"WARNING: skipping {fname}: {e}")
    return docs

def chunk_document(text: str, source: str,
                   doc_type: str) -> list[dict]:
    if doc_type == "structured":
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [{"text": json.dumps(item),
                         "source": source,
                         "chunk_index": i}
                        for i, item in enumerate(data)]
        except:
            pass
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    i = 0
    idx = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + CHUNK_SIZE]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append({"text": chunk_text,
                       "source": source,
                       "chunk_index": idx})
        i += CHUNK_SIZE - CHUNK_OVERLAP
        idx += 1
    return chunks

def build_index():
    print(f"Loading corpus from: "
          f"{os.path.abspath(CORPUS_PATH)}")
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        db.delete_collection("scaler_rag")
    except:
        pass
    collection = db.create_collection(
        "scaler_rag",
        metadata={"hnsw:space": "cosine"}
    )
    docs = load_documents(CORPUS_PATH)
    print(f"Found {len(docs)} documents")
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc["text"],
                               doc["source"],
                               doc["doc_type"])
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"Embedding {len(all_chunks)} chunks...")
    batch_size = 10
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(texts)
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"source": c["source"],
                       "chunk_index": c["chunk_index"]}
                      for c in batch],
            ids=[f"chunk_{i+j}"
                 for j in range(len(batch))]
        )
        print(f"  {min(i+batch_size, len(all_chunks))}"
              f"/{len(all_chunks)} embedded")

    print(f"\n✅ Done: {len(all_chunks)} chunks "
          f"from {len(docs)} docs")
    return collection

if __name__ == "__main__":
    build_index()