from fastapi import APIRouter, Header, HTTPException
import hmac
import os

import openai

from ..controllers import AgenticController
from ..interfaces import MessagePayload, MemberMessagePayload, RagMessagePayload
from ..llm import LlmNotConfigured

router = APIRouter(prefix="/chat-ai")

agenticController = AgenticController()

UNAVAILABLE = 'Trợ lý AI đang tạm thời không khả dụng. Vui lòng thử lại sau ít phút.'


def require_internal_token(x_internal_token: str | None):
    expected = os.getenv('AGENTIC_INTERNAL_TOKEN', '').strip()
    if not expected:
        raise HTTPException(status_code=503, detail='AI internal authentication is not configured.')
    if not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
        raise HTTPException(status_code=401, detail='Invalid internal token.')


def require_bearer(authorization: str | None):
    if not authorization or not authorization.lower().startswith('bearer '):
        raise HTTPException(status_code=401, detail='Missing authorization.')


async def guarded(label: str, call):
    """Maps provider failures to stable, non-leaky HTTP errors."""
    try:
        return await call()
    except HTTPException:
        raise
    except LlmNotConfigured:
        raise HTTPException(status_code=503, detail='Trợ lý AI chưa được cấu hình.')
    except openai.RateLimitError:
        raise HTTPException(status_code=503, detail='Trợ lý AI đang quá tải. Vui lòng thử lại sau ít phút.')
    except (openai.AuthenticationError, openai.PermissionDeniedError) as error:
        print(f'[{label}] provider rejected credentials: {type(error).__name__}')
        raise HTTPException(status_code=503, detail=UNAVAILABLE)
    except (openai.APITimeoutError, openai.APIConnectionError) as error:
        print(f'[{label}] provider unreachable: {type(error).__name__}')
        raise HTTPException(status_code=504, detail='Trợ lý AI phản hồi quá lâu. Vui lòng thử lại.')
    except Exception as error:
        print(f'[{label}] error: {type(error).__name__}: {str(error)[:300]}')
        raise HTTPException(status_code=502, detail=UNAVAILABLE)


@router.post("/")
async def staff_chat(payload: MessagePayload, authorization: str | None = Header(default=None), x_internal_token: str | None = Header(default=None)):
    """Admin / expert copilot: LLM + allow-listed tools over the BFF."""
    require_internal_token(x_internal_token)
    require_bearer(authorization)
    return await guarded('chat-ai', lambda: agenticController.staffChat(
        payload.message, authorization, payload.role, payload.user_id or payload.admin_id, payload.name,
        [turn.model_dump() for turn in payload.history]))


@router.post("/member")
async def member_chat(payload: MemberMessagePayload, authorization: str | None = Header(default=None), x_internal_token: str | None = Header(default=None)):
    """Student assistant: career advice, latest test result, expert suggestions, booking help, site FAQ."""
    require_internal_token(x_internal_token)
    require_bearer(authorization)
    return await guarded('chat-ai/member', lambda: agenticController.memberChat(
        payload.message, authorization, payload.user_id, payload.name,
        [turn.model_dump() for turn in payload.history]))


@router.post("/rag")
async def chat_with_rag(payload: RagMessagePayload, x_internal_token: str | None = Header(default=None)):
    """Embedding → Qdrant search → LLM answer (RAG pipeline)"""
    require_internal_token(x_internal_token)
    return await guarded('chat-ai/rag', lambda: agenticController.callAiFromClient(
        payload.message, [turn.model_dump() for turn in payload.history]))
