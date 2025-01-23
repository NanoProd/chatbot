#models/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    user_input: str
    model: str = "gpt-3.5-turbo"
    conversation_id: Optional[int] = None
    use_reranking: bool = False

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

class SearchResult(BaseModel):
    topic: str
    text: str
    relevance_score: float