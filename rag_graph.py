import json
import os
from typing import TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, START, END

import vector_store as vs

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

NOT_FOUND_MSG = "I couldn't find this in the document."

_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


SYSTEM_PROMPT = f"""You are a question-answering assistant for the "Agentic AI" \
eBook. You must answer using ONLY the information inside the <context> block \
provided in the user message. Do not use any outside knowledge, prior \
training data, or assumptions, even if you are confident about the answer.

Rules:
- If the answer is fully or partially supported by the context, answer it \
concisely and cite nothing beyond what's given — just answer in plain prose.
- If the context does not contain enough information to answer the \
question, you MUST respond with exactly this sentence and nothing else: \
"{NOT_FOUND_MSG}"
- Never fill gaps in the context with general knowledge about AI, agents, \
or any other topic, even if it seems obviously true.
"""


class RAGState(TypedDict, total=False):
    query: str
    retrieved_chunks: list[dict]   
    graded_chunks: list[dict]      
    final_answer: str
    confidence: float


def retrieve_node(state: RAGState) -> RAGState:
    hits = vs.similarity_search(state["query"], k=TOP_K)
    return {"retrieved_chunks": hits}


GRADE_PROMPT = """You are a strict relevance grader. Given a user question \
and a numbered list of text passages, return ONLY a JSON array of the \
numbers of the passages that contain information directly useful for \
answering the question. If none are useful, return an empty array [].

Question: {question}

Passages:
{passages}

Respond with ONLY the JSON array, e.g. [0, 2]. No other text."""


def grade_node(state: RAGState) -> RAGState:
    chunks = state["retrieved_chunks"]
    if not chunks:
        return {"graded_chunks": []}

    if _client is None:
        graded = [c for c in chunks if c["score"] >= 0.3]
        return {"graded_chunks": graded}

    passages_block = "\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks))
    try:
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "user", "content": GRADE_PROMPT.format(
                    question=state["query"], passages=passages_block)}
            ],
            temperature=0,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()
    
        raw = raw[raw.find("[") : raw.rfind("]") + 1]
        relevant_indices = set(json.loads(raw))
    except Exception:
      relevant_indices = {i for i, c in enumerate(chunks) if c["score"] >= 0.3}

    graded = [c for i, c in enumerate(chunks) if i in relevant_indices]
    return {"graded_chunks": graded}


def has_relevant_chunks(state: RAGState) -> str:
    """Conditional edge: route to `generate` or `fallback`."""
    return "generate" if state.get("graded_chunks") else "fallback"


def generate_node(state: RAGState) -> RAGState:
    chunks = state["graded_chunks"]
    context = "\n\n".join(
        f"[Page {c['page']}] {c['text']}" for c in chunks
    )
    user_message = f"<context>\n{context}\n</context>\n\nQuestion: {state['query']}"

    if _client is None:
        return {
            "final_answer": "GROQ_API_KEY is not set — cannot generate an answer.",
            "confidence": 0.0,
        }

    response = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,  
        max_tokens=500,
    )
    answer = response.choices[0].message.content.strip()


    top_score = max(c["score"] for c in chunks)
    confidence = max(0.0, min(1.0, top_score))

    return {"final_answer": answer, "confidence": confidence}


def fallback_node(state: RAGState) -> RAGState:
    return {"final_answer": NOT_FOUND_MSG, "confidence": 0.0}


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        has_relevant_chunks,
        {"generate": "generate", "fallback": "fallback"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()


_graph = build_graph()


def run_query(query: str) -> dict:
    """Run the full graph for a single query and return API/UI-ready output."""
    result = _graph.invoke({"query": query})


    context_chunks = [
        {"text": c["text"], "page": c["page"], "score": round(c["score"], 4)}
        for c in result.get("retrieved_chunks", [])
    ]

    return {
        "answer": result["final_answer"],
        "context_chunks": context_chunks,
        "confidence": round(result.get("confidence", 0.0), 4),
    }


if __name__ == "__main__":

    import sys

    q = " ".join(sys.argv[1:]) or "What is Agentic AI?"
    out = run_query(q)
    print(json.dumps(out, indent=2))
