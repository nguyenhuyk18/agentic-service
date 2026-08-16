from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from  apps.routers import ClientRouter
from apps.routers import CareerRouter
from dotenv import load_dotenv
import os
from pathlib import Path
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # cho tất cả origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env_path = (
    Path(__file__)
    .resolve()
    .parent / '.env'
)

load_dotenv(env_path)

app.include_router(ClientRouter.router)
app.include_router(CareerRouter.router)

print(f"Groq model: {os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}")

class RequestData(BaseModel):
    text: str

@app.get('/')
def home ():
    return {
        "message" : "Hello chào bạn đã đến đây hehehehehehehehehe"
    }

@app.post('/embedding')
def embedding(data: RequestData):
    from apps.controllers.AgenticController import get_embedding_model
    vector = get_embedding_model().encode(data.text).tolist()
    return { "vector" : vector }

