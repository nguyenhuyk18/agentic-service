from pydantic import BaseModel, Field
from typing import Literal


class ChatTurn(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(min_length=1, max_length=1500)


class MessagePayload(BaseModel):
    """Staff copilot. role/user_id/name are set by the BFF from the verified token."""
    message: str = Field(min_length=1, max_length=2000)
    admin_id: str | None = None
    role: Literal['admin', 'expert'] = 'admin'
    user_id: str | None = Field(default=None, max_length=64)
    name: str | None = Field(default=None, max_length=120)
    history: list[ChatTurn] = Field(default_factory=list, max_length=10)


class MemberMessagePayload(BaseModel):
    message: str = Field(min_length=1, max_length=1500)
    user_id: str = Field(min_length=1, max_length=64)
    name: str | None = Field(default=None, max_length=120)
    history: list[ChatTurn] = Field(default_factory=list, max_length=10)


class RagMessagePayload(BaseModel):
    message: str = Field(min_length=1, max_length=1500)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)
