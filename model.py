# model.py

from sentence_transformers import SentenceTransformer, util
import torch

# Load model once on import
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_similarity(text1: str, text2: str) -> float:
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return round(similarity, 4)