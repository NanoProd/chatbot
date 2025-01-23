from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from settings import settings
from database.models import Base
from database.connection import engine, get_db
from api.chat_handler import ChatHandler
from models.schemas import ChatRequest, ChatResponse
from pinecone import Pinecone

try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Error creating tables: {str(e)}")

app = FastAPI(debug=True)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index("medical-index")
        chat_handler = ChatHandler(db, index)
        response, conversation_id, sources = await chat_handler.generate_response(
            request.user_input,
            use_reranking=request.use_reranking
        )

        return ChatResponse(
            response=response,
            sources=[f"Source: {s}" for s in sources]
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))