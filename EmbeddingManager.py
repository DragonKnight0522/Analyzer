from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text):
        return self.model.encode(text, convert_to_tensor=True)
