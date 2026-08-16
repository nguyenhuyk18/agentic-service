from fastapi import APIRouter, HTTPException
from ..controllers import AgenticController

from ..interfaces import MessagePayload

router = APIRouter(prefix="/chat-ai")

agenticController = AgenticController()


@router.post("/")
async def get_users(payload: MessagePayload):
    """Dùng LLM + Tools (email, list expert) — không qua RAG"""
    try:
        return await agenticController.pickToolsFunc(payload.message)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except Exception as error:
        print(f"[chat-ai] Groq error: {error}")
        raise HTTPException(status_code=502, detail="Không thể kết nối Groq Cloud lúc này.")


@router.post("/rag")
async def chat_with_rag(payload: MessagePayload):
    """Embedding → Qdrant search → LLM answer (RAG pipeline)"""
    try:
        return await agenticController.callAiFromClient(payload.message)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except Exception as error:
        print(f"[chat-ai/rag] Groq error: {error}")
        raise HTTPException(status_code=502, detail="Không thể kết nối Groq Cloud lúc này.")
