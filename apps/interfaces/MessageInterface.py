from pydantic import BaseModel, Field
from typing import Literal

class MessagePayload(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    admin_id: str | None = None

class ChatTurn(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(min_length=1, max_length=1200)

class RagMessagePayload(BaseModel):
    message: str = Field(min_length=1, max_length=1500)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)
