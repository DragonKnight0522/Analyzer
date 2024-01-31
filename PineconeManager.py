import pinecone

class PineconeManager:
    def __init__(self, api_key, environment):
        self.api_key = api_key
        self.environment = environment
        self.index = None

    def initialize_pinecone(self, index_name):
        pinecone.init(api_key=self.api_key, environment=self.environment)
        self.index = pinecone.Index(index_name)

    def store_embedding(self, location_id, embedding):
        try:
            self.index.upsert(vectors=[(location_id, embedding)])
            print(f'Embedding for location {location_id} stored in Pinecone.')
        except Exception as e:
            print(f'Error storing embedding in Pinecone: {e}')
            raise e
