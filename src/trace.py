import json
import time
from datetime import datetime
from config import TRACE_FILE

def emit_trace(query: str, chunks: list[dict],
               result: dict, latency_ms: float):
    trace = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "latency_ms": round(latency_ms, 2),
        "top_k_chunks": [{
            "source": c["source"],
            "similarity": c["similarity"],
            "text_preview": c["text"][:200]
        } for c in chunks],
        "top_similarity": chunks[0]["similarity"]
                          if chunks else 0,
        "no_match_triggered": result.get("no_match", False),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "answer_preview": result["answer"][:300]
    }

    with open(TRACE_FILE, "a") as f:
        f.write(json.dumps(trace) + "\n")

    print(f"[TRACE] latency={latency_ms:.0f}ms | "
          f"top_similarity={trace['top_similarity']} | "
          f"tokens={trace['prompt_tokens']}")

    return trace
