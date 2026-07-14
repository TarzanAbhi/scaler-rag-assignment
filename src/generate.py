import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import anthropic
from config import SIMILARITY_THRESHOLD

SCALER_KEY = os.environ.get("SCALER_LLM_API_KEY")
_client = anthropic.Anthropic(api_key=SCALER_KEY)

SYSTEM_PROMPT = """You are a helpful assistant for \
Scaler's learner support team.
Answer using ONLY the context provided below.
If context is insufficient say exactly:
"I don't have enough information to answer this \
accurately based on the available documents."
Do not make up facts. Cite sources at the end like:
Sources: [filename.txt]"""

def generate_answer(query: str,
                    chunks: list[dict]) -> dict:
    if not chunks or \
       chunks[0]["similarity"] < SIMILARITY_THRESHOLD:
        return {
            "answer": "No relevant information found "
                     "in the knowledge base.",
            "no_match": True,
            "prompt": None,
            "prompt_tokens": 0,
            "completion_tokens": 0
        }

    context = "\n\n---\n\n".join([
        f"Source: {c['source']}\n{c['text']}"
        for c in chunks
    ])

    user_message = f"""Context:
{context}

Question: {query}

Answer:"""

    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user",
                   "content": user_message}]
    )

    return {
        "answer": response.content[0].text,
        "no_match": False,
        "prompt": user_message,
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens
    }
