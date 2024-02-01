from src.DatabaseManager import DatabaseManager
from src.AIModel import AIModel

class Analyzer:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.ai_model = AIModel()

    def load_data(self):
        self.db_manager.connect_to_database()
        return self.db_manager.get_locations()

    def analyze_locations(self, locations):
        if not locations:
            print('No locations found to analyze.')
            return
        
        for location in locations:
            try:
                location_id = location[0]
                embedding = self.ai_model.generate_embedding(location)
                self.ai_model.store_embedding(embedding)
                criteria = self.ai_model.extract_criteria()
                self.db_manager.store_criteria(location_id, criteria)
            except Exception as e:
                print(f'Error processing location {location_id}: {e}')
