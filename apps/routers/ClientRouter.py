from fastapi import APIRouter, Header, HTTPException
from ..controllers import AgenticController

from ..interfaces import MessagePayload, RagMessagePayload
import hmac
from groq import PermissionDeniedError
import os

router = APIRouter(prefix="/chat-ai")

agenticController = AgenticController()


@router.post("/")
async def get_users(payload: MessagePayload, authorization: str | None = Header(default=None), x_internal_token: str | None = Header(default=None)):
    """Dùng LLM + Tools (email, list expert) — không qua RAG"""
    try:
        expected = os.getenv('AGENTIC_INTERNAL_TOKEN', '').strip()
        if not expected:
            raise HTTPException(status_code=503, detail='AI internal authentication is not configured.')
        if not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
            raise HTTPException(status_code=401, detail='Invalid internal token.')
        if not authorization or not authorization.lower().startswith('bearer '):
            raise HTTPException(status_code=401, detail='Missing authorization.')
        return await agenticController.pickToolsFunc(payload.message, authorization, payload.admin_id)
    except PermissionDeniedError:
        raise HTTPException(status_code=503, detail='Model AI đang bị tổ chức Groq chặn. Quản trị viên cần bật quyền cho model hoặc cấu hình model được phép sử dụng.')
    except HTTPException:
        raise
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except Exception as error:
        print(f"[chat-ai] Groq error: {error}")
        raise HTTPException(status_code=502, detail="Không thể kết nối Groq Cloud lúc này.")


@router.post("/rag")
async def chat_with_rag(payload: RagMessagePayload, x_internal_token: str | None = Header(default=None)):
    """Embedding → Qdrant search → LLM answer (RAG pipeline)"""
    try:
        expected = os.getenv('AGENTIC_INTERNAL_TOKEN', '').strip()
        if not expected:
            raise HTTPException(status_code=503, detail='RAG internal authentication is not configured.')
        if not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
            raise HTTPException(status_code=401, detail='Invalid internal token.')
        return await agenticController.callAiFromClient(payload.message, [turn.model_dump() for turn in payload.history])
    except PermissionDeniedError:
        raise HTTPException(status_code=503, detail='Model AI đang bị tổ chức Groq chặn. Quản trị viên cần bật quyền cho model hoặc cấu hình model được phép sử dụng.')
    except HTTPException:
        raise
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except Exception as error:
        print(f"[chat-ai/rag] Groq error: {error}")
        raise HTTPException(status_code=502, detail="Không thể kết nối Groq Cloud lúc này.")
