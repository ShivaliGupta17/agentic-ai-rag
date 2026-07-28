from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_graph import run_query

app = FastAPI(
    title="Agentic AI eBook RAG Chatbot",
    description="Answers questions strictly from the 'Agentic AI' eBook via a "
                "LangGraph retrieve -> grade -> generate pipeline.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")


class ContextChunk(BaseModel):
    text: str
    page: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    context_chunks: list[ContextChunk]
    confidence: float


@app.get("/")
def health_check():
    return {"status": "ok", "message": "POST /chat with {'query': '...'} to ask a question."}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = run_query(request.query)
    return ChatResponse(**result)
