import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import time
import argparse
from ingest import build_index
from retrieve import retrieve, get_collection
from generate import generate_answer
from trace import emit_trace
import inspect

print(inspect.signature(emit_trace))
def ask(query: str, collection=None) -> str:
    start = time.time()
    chunks = retrieve(query, collection)
    result = generate_answer(query, chunks)
    latency_ms = (time.time() - start) * 1000
    emit_trace(query, chunks, result, latency_ms)
    return result["answer"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str,
                        help="Question to ask")
    parser.add_argument("--ingest", 
                        action="store_true",
                        help="Rebuild the index")
    args = parser.parse_args()
    
    if args.ingest:
        build_index()
        return
    
    if args.query:
        collection = get_collection()
        answer = ask(args.query, collection)
        print(f"\nAnswer: {answer}")
    else:
        # Interactive mode
        print("Scaler RAG Q&A — type 'exit' to quit\n")
        collection = get_collection()
        while True:
            query = input("Question: ").strip()
            if query.lower() == "exit":
                break
            if query:
                answer = ask(query, collection)
                print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()