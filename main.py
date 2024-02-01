from src.Analyzer import Analyzer
from dotenv import load_dotenv

load_dotenv()

# Instantiate the Analyzer class
analyzer = Analyzer()

# Load data from the database
locations = analyzer.load_data()

# Analyze locations
analyzer.analyze_locations(locations)