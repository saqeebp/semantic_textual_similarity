import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticModel:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    def predict(self, text1: str, text2: str) -> float:
        embeddings = self.model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float((similarity + 1) / 2)
