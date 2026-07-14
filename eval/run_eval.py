import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import anthropic

from config import OPENAI_API_KEY, GENERATION_MODEL
from retrieve import retrieve, get_collection
from generate import generate_answer

from generate import _client


def llm_judge(query: str, answer: str,
              chunks: list[dict], judge_type: str) -> dict:
    context = "\n\n".join([c["text"] for c in chunks])
    sources = [c["source"] for c in chunks]

    if judge_type == "faithfulness":
        prompt = f"""You are an evaluation judge.

Determine if the answer is faithful to the context.
Faithful means the answer does not contradict or go
beyond what the context says.

Question: {query}

Context:
{context}

Answer:
{answer}

Respond with ONLY valid JSON, no markdown, no backticks:

{{"faithful": true, "reason": "one sentence"}}
"""

    else:
        prompt = f"""You are an evaluation judge.

Determine if the retrieved sources are relevant and
likely contain the answer to the question.

Question: {query}

Retrieved sources:
{sources}

Answer:
{answer}

Respond with ONLY valid JSON, no markdown, no backticks:

{{"context_precise": true, "reason": "one sentence"}}
"""

    response = _client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=200,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    try:
        text = response.content[0].text.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end != 0:
            text = text[start:end]

        return json.loads(text)

    except Exception:
        if judge_type == "faithfulness":
            return {
                "faithful": False,
                "reason": "Judge parse error"
            }

        return {
            "context_precise": False,
            "reason": "Judge parse error"
        }


def run_eval():
    test_cases_path = os.path.join(
        os.path.dirname(__file__), "test_cases.json"
    )

    with open(test_cases_path, "r") as f:
        test_cases = json.load(f)

    collection = get_collection()

    results = []
    passed = 0
    faithful_count = 0
    precise_count = 0

    print(f"\nRunning {len(test_cases)} test cases...\n")
    print("-" * 60)

    for tc in test_cases:
        query = tc["query"]

        chunks = retrieve(query, collection)
        result = generate_answer(query, chunks)
        answer = result["answer"]
        answer_lower = answer.lower()

        # Keyword evaluation
        if tc.get("is_no_match"):
            keyword_pass = any(
                phrase in answer_lower
                for phrase in [
                    "don't have",
                    "not found",
                    "no relevant",
                    "no information",
                    "accurately based",
                ]
            )
        else:
            keyword_pass = any(
                kw.lower() in answer_lower
                for kw in tc["expected_keywords"]
            )

        faithful = True
        precise = True
        faith_reason = "N/A (no-match case)"
        precise_reason = "N/A (no-match case)"

        if not tc.get("is_no_match") and chunks:

            faith_result = llm_judge(
                query,
                answer,
                chunks[:3],
                "faithfulness",
            )

            faithful = faith_result.get("faithful", False)
            faith_reason = faith_result.get("reason", "")

            if faithful:
                faithful_count += 1

            prec_result = llm_judge(
                query,
                answer,
                chunks[:3],
                "context_precision",
            )

            precise = prec_result.get("context_precise", False)
            precise_reason = prec_result.get("reason", "")

            if precise:
                precise_count += 1

        case_pass = keyword_pass and faithful

        if case_pass:
            passed += 1

        status = "✅ PASS" if case_pass else "❌ FAIL"

        print(f"{status} [{tc['id']}] {query}")
        print(
            f"       Keywords: {'✅' if keyword_pass else '❌'} | "
            f"Faithful: {'✅' if faithful else '❌'} | "
            f"Precise: {'✅' if precise else '❌'}"
        )

        if not faithful:
            print(f"       Faithfulness: {faith_reason}")

        if not precise:
            print(f"       Precision: {precise_reason}")

        print()

        results.append({
            "id": tc["id"],
            "query": query,
            "answer": answer,
            "keyword_pass": keyword_pass,
            "faithful": faithful,
            "context_precise": precise,
            "faith_reason": faith_reason,
            "precise_reason": precise_reason,
            "passed": case_pass
        })

    non_no_match = [
        tc for tc in test_cases
        if not tc.get("is_no_match")
    ]

    total_non_match = len(non_no_match)

    print("-" * 60)
    print(f"Results:            {passed}/{len(test_cases)} passed")
    print(
        f"Faithfulness:       "
        f"{faithful_count}/{total_non_match} faithful"
    )
    print(
        f"Context Precision:  "
        f"{precise_count}/{total_non_match} precise"
    )

    out_path = os.path.join(
        os.path.dirname(__file__),
        "results.json",
    )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nDetailed results saved to eval/results.json")


if __name__ == "__main__":
    run_eval()