from DatabaseManager import DatabaseManager
from EmbeddingManager import EmbeddingManager
from PineconeManager import PineconeManager
from CriteriaExtractor import CriteriaExtractor

class Analyzer:
    def __init__(self, db_params, embedding_model, pinecone_api_key, pinecone_env, openai_api_key):
        self.db_manager = DatabaseManager(db_params)
        self.embedding_manager = EmbeddingManager(embedding_model)
        self.pinecone_manager = PineconeManager(pinecone_api_key, pinecone_env)
        self.criteria_extractor = CriteriaExtractor(openai_api_key)

    def load_data(self):
        self.db_manager.connect_to_database()
        return self.db_manager.get_locations()

    def analyze_locations(self, locations):
        if not locations:
            print('No locations found to analyze.')
            return
        self.pinecone_manager.initialize_pinecone('location_index')
        self.criteria_extractor.initialize_chain()
        for location in locations:
            try:
                location_id, location_info = location
                embedding = self.embedding_manager.generate_embedding(location_info)
                self.pinecone_manager.store_embedding(location_id, embedding)
                prompt = f'Extract criteria for the following location information: {location_info}'
                criteria = self.criteria_extractor.extract_criteria(prompt)
                self.db_manager.store_criteria(location_id, criteria)
            except Exception as e:
                print(f'Error processing location {location_id}: {e}')
