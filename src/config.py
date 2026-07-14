import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or \
                 os.environ.get("SCALER_LLM_API_KEY")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or \
                 os.environ.get("SCALER_LLM_API_KEY")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.0-flash"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3
CHROMA_PATH = "./index"
CORPUS_PATH = "./corpus"
TRACE_FILE = "./traces.jsonl"
