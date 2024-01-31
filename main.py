from Analyzer import Analyzer

# Sample environment variables
DB_PARAMS = {
    'host': 'localhost',
    'database': 'locations_db',
    'user': 'user',
    'password': 'password'
}
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
PINECONE_API_KEY = 'sample-pinecone-key'
PINECONE_ENV = 'us-west1-gcp'
OPENAI_API_KEY = 'sample-openai-key'

# Instantiate the Analyzer class
analyzer = Analyzer(DB_PARAMS, EMBEDDING_MODEL, PINECONE_API_KEY, PINECONE_ENV, OPENAI_API_KEY)

# Load data from the database
locations = analyzer.load_data()

# Analyze locations
analyzer.analyze_locations(locations)