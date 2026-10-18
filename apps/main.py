from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi import Header, HTTPException
from  apps.routers import ClientRouter
from apps.routers import CareerRouter
from apps.routers import CareerModelRouter
from dotenv import load_dotenv
import os
from pathlib import Path
app = FastAPI(docs_url=None if os.getenv('APP_ENV') == 'production' else '/docs')

# CORS middleware
allowed_origins = [origin.strip() for origin in os.getenv('CORS_ALLOWED_ORIGINS', 'http://localhost:3000').split(',') if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StripPathPrefix:
    """ALB forwards /ai/... unchanged (Caddy used to strip it), so drop the prefix here."""

    def __init__(self, app, prefix: str):
        self.app = app
        self.prefix = prefix

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] in ("http", "websocket") and (path == self.prefix or path.startswith(self.prefix + "/")):
            scope = dict(scope, path=path[len(self.prefix):] or "/")
        await self.app(scope, receive, send)


app.add_middleware(StripPathPrefix, prefix="/ai")

env_path = (
    Path(__file__)
    .resolve()
    .parent / '.env'
)

load_dotenv(env_path)

app.include_router(ClientRouter.router)
app.include_router(CareerRouter.router)
app.include_router(CareerModelRouter.router)

print(f"Groq model: {os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')}")

class RequestData(BaseModel):
    text: str = Field(min_length=1, max_length=1000)

@app.get('/')
def home ():
    return {
        "message" : "MajorGuide Agentic AI service"
    }

@app.post('/embedding')
async def embedding(data: RequestData, x_internal_token: str | None = Header(default=None)):
    import hmac
    import asyncio
    expected = os.getenv('AGENTIC_INTERNAL_TOKEN', '').strip()
    if not expected or not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
        raise HTTPException(status_code=401, detail='Invalid internal token')
    from apps.controllers.AgenticController import get_embedding_model
    vector = (await asyncio.to_thread(get_embedding_model().encode, data.text)).tolist()
    return { "vector" : vector }

