
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.torchserve_client import get_ner
from app.services.mistral_client import chat_with_mistral

router = APIRouter()

class NERRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    message: str
    history: list[str] = []

#NER (Named Entity Recognition)
@router.post("/ner")
def ner_endpoint(payload: NERRequest):
    try:
        return {"entities": get_ner(payload.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
def chat_endpoint(payload: ChatRequest):
    try:
        response = chat_with_mistral(payload.message, payload.history)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health():
    return {"status": "ok"}
