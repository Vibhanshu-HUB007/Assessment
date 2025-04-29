# app.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from model import get_similarity

app = FastAPI()

class TextPair(BaseModel):
    text1: str
    text2: str

@app.post("/")
async def compute_similarity(payload: TextPair):
    score = get_similarity(payload.text1, payload.text2)
    return {"similarity score": score}