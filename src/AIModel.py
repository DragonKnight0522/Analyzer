import os
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from pinecone import Pinecone
from langchain_community.vectorstores.pinecone import Pinecone as LLMPinecone
from langchain_openai import ChatOpenAI

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class AIModel:
    def __init__(self):
        self.txt_splitter = RecursiveCharacterTextSplitter(chunk_size=int(os.environ.get("TEXT_SPLITTER_CHUNK_SIZE")), chunk_overlap=int(os.environ.get("TEXT_SPLITTER_CHUNK_OVERLAP")))
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = SentenceTransformer(os.environ.get("EMBEDDING_MODEL"), device=device)

        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index = pc.Index(os.environ.get("PINECONE_INDEX_NAME"))

        self.chain = None
        self.llm = ChatOpenAI(model_name=os.environ["OPENAI_API_MODEL"], openai_api_key=os.environ["OPENAI_API_KEY"])
        template = """Answer the question based only on the following context:
        {context}

        - Your Role:
            * You are a highly skilled data analyst tasked with interpreting and analyzing location-based information. 
            * Your goal is to deduce and suggest at least one option mentioned in predefined list for each criterion based on the location data provided, even if specific services are not explicitly mentioned. 
            * You may use educated guesses based on the nature of the location, if the options are not explicitly mentioned.
            * the options must exist in criteria's predefined list.
        - Criteria's predefined list (ensure at least one selection per criterion): 
            barType: ["Dive Bar", "Pub", "Club", "Wine Bar", "Beer Hall/Garden", "Cocktail Bar", "Lounge", "Sports Bar", "Speakeasy", "Distillery", "Brewery"],
            specialtyType: ["Golf", "Arcade", "Bowling Alley", "Pool Hall", "Music Venue", "Comedy Club", "Karaoke Bar", "Axe Throwing", "Strip Club", "Piano Bar"],
            events: ["Comedy Night", "Trivia Night", "Karaoke Night", "Live Music", "Board Game Night", "Ladies Night", "Open Mic Night"],
            games: ["Darts", "Pool", "Shuffle Board", "Beer Pong", "Jenga", "Pingpong", "Arcade Games", "Cornhole", "Volleyball", "Boardgames", "Skeeball", "Pinball", "Air hockey", "Big Connect 4", "None"],
            barMusic: ["EDM", "Country", "Modern", "Alternative", "Throwbacks", "Hiphop", "Rock", "Jazz", "Acoustic", "Blues", "Pop", "Piano"],
            parking: ["1", "2", "3", "4", "5"],
            dogFriendly: "Yes" or "No",
            time2visit: ["Day Drinking (12pm - 4pm)", "Brunch (6am - 12pm)", "Happy Hour (4pm - 6pm)", "Pregame (6pm - 9pm)", "Night Scene (9pm-12am)", "Late Night (12am-close)"],
            amenities: ["Heated Patio", "Dance Floor", "Patio", "Rooftop", "Lots of Seating", "PhotoBooth", "Food Trucks", "Lots of Tvs", "Big Screen Tv"],
            ambiance: ["Party", "Intimate", "Mellow", "College Bar", "Lounge", "Casual", "Formal", "Cozy", "LGBTQ", "Views", "Busy", "Calm", "Spacious", "Unique", "Holiday", "Upscale", "Themed", "Dive"],
            close2Others: ["1", "2", "3", "4", "5"],
            sports: ["1", "2", "3", "4", "5", "None"],
            typesOfSports: ["Football", "Soccer", "Basketball", "MMA", "Boxing", "Golf", "Tennis", "Baseball", "Hockey"],
            dancing: "Yes" or "No",
            mixology: ["1", "2", "3", "4", "5", "None"],
            drinkCost: ["0-5", "8-11", "5-8", "12-15", "15-18", "19+", "None"],
            beerCost: ["3-5", "5-7", "7-10", "10+", "None"],
            drinkSpecialties: ["Margaritas", "Shots", "Cocktails", "Sake", "Wine", "Craft Beer", "Brunch Drinks", "Cider", "Seltzer"],
            happyHour: ["Food HH", "Drink HH", "Late Night HH", "Weekday HH", "All Week HH"],
            offersFood: "Yes" or "No",
            foodType: ["Afghan", "African", "Albanian", "American", "Arabian", "Argentinian", "Asian", "Australian", "BBQ", "Bagels", "Bakery", "Bangladeshi", "Belgian", "Brazilian", "Breakfast", "British", "Brunch", "Burmese", "Burritos", "Cajun", "Californian", "Calzone", "Canadian", "Cantonese", "Caribbean", "CheeseSteaks", "Chicken", "Chilean", "Chili", "Chinese", "Classic", "Coffee and Tea", "Colombian", "Creole", "Crepes", "Cuban", "Deli", "Dessert", "Dim Sum", "Diner", "Dinner", "Dominican", "Eclectic", "Ecuadorian", "Egyptian", "El Salvadoran", "Empanada", "English", "Ethiopian", "Filipino", "Fine Dining", "French", "Frozen Yogurt", "German", "Gluten-Free", "Greek", "Guatemalan", "Gyro", "Haitian", "Halal", "Hamburgers", "Hawaiian", "Healthy", "Hoagies", "Hot Dogs", "Ice Cream", "Indian", "Indonesian", "Irish", "Israeli", "Italian", "Jamaican", "Japanese", "Jewish", "Keto", "Korean", "Kosher", "Latin American", "Lebanese", "Low Carb", "Low Fat", "Malaysian", "Mediterranean", "Mexican", "Middle Eastern", "Mongolian", "Moroccan", "Nepalese", "New American", "Nicaraguan", "Nigerian", "Noodles", "Organic", "Pakistani", "Panamanian", "Pasta", "Persian", "Peruvian", "Pitas", "Pizza", "Poke", "Polish", "Portuguese", "Ramen", "Ribs", "Russian", "Salads", "Sandwiches", "Scandinavian", "Seafood", "Senegalese", "Shakes", "Smoothies and Juices", "Soul Food", "Soup", "Southern", "Southwestern", "Spanish", "Steak", "Subs", "Sushi", "Swedish", "Tacos", "Taiwanese", "Tapas", "Thai", "Tibetan", "Turkish", "Ukrainian", "Vegan", "Vegetarian", "Venezuelan", "Vietnamese", "Wings", "Wraps"],
            foodCost: ["1", "2", "3", "4"],
            restaurantType: ["Diner", "Bistro", "Steakhouse", "Tavern", "Cafe", "Fine Dining", "Casual Dinning", "Gastropub", "Bakery", "Buffet", "Food Hall", "Food Truck", "Fast Casual"],
        - Description of each criterion:
            barType: "This criterion indicates the bar type of at this location.",
            specialtyType: "This criterion indicates the type of specialties served at this location.",
            events: "This criterion indicates the type of events organized at this location.",
            games: "This criterion indicates the type of games played in this location.",
            barMusic: "This criterion indicates the type of musics the user can hear in this location.",
            parking: "This criterion indicates how easy to park in this location.",
            dogFriendly: "This criterion indicates whether this location is pet friendly.",
            time2visit: "This criterion indicates the best time of visit to this location.",
            amenities: "This criterion indicates the type of amenities available at this location.",
            ambiance: "This criterion indicates the type of ambiances for this location.",
            close2Others: "This criterion indicates how close this location to other locations.",
            sports: "This criterion indicates how good to watching sports.",
            typesOfSports: "This criterion indicates the types of sports user can watch.",
            dancing: "This criterion indicates whether there is dance.",
            mixology: "This criterion indicates the quality of mixology served at this location.",
            drinkCost: "This criterion indicates the price range of drink.",
            beerCost: "This criterion indicates the price range of beer.",
            drinkSpecialties: "This criterion indicates the type of drink specials served at this location.",
            happyHour: "This criterion indicates the type of happy hour offered at this location.",
            offersFood: "This criterion indicates whether the location offers food.",
            foodType: "This criterion indicates the type of food served at this location.",
            foodCost: "This criterion indicates the price range of food served at this location.",
            restaurantType: "This criterion indicates the type of restaurant at this location.",
        - Output:
            * Generate a list of criteria and their options based on the provided location information. Your output should include at least one option for each criteria, deducing possibilities from the available data.
            * Sample output type: 
                "criteria 1": ["option 1", ...],
                "criteria 2": ["option 1", ...],
                ...
        - Instructions for Selecting Options:
            * Analyze the provided location information to deduce services that could align with the predefined criteria.
            * Each criteria's value should contain at least one applicable option, making educated guesses if necessary, existing in predefined criteria.
            * Analysis data is the result of previous analysis, so please avoid duplication and accumulate it.
            * If there is no "None" option in each criterion, then must select at least one value based on location information and educated guesses of location.
        - Rules for Response:
            * Your response must be a JSON object containing all relevant criteria as keys.
            * Do not include backticks, help text, explanations, dots, or code syntax in responses.
            * Strictly adhere to the predefined criteria's options. Select at least one option for each criteria, using educated guesses based on the location data.
            * Your response should not include options that do not exist in the predefined criteria's options.
            * Avoid including activities or services not specifically listed in the predefined criteria.
        Question: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(template)

    def generate_embedding(self, info):
        _, yelp_feature, yelp_amenities, yelp_about, yelp_menu, yelp_name, yelp_reviews, data = info
        texts = [
            f'feature of {yelp_name} are {yelp_feature}',
            f'amenities of {yelp_name} are {yelp_amenities}',
            f"about of {yelp_name} are {yelp_about}",
            f'menu of {yelp_name} are {yelp_menu}',
            f'reviews of {yelp_name} are {yelp_reviews}',
            # f'website data of {yelp_name} are {data}',
        ]
        chunked_data = []
        for text in texts:
            print(text)
            chunked_data.extend(self.txt_splitter.split_text(text))

        embeddings = self.model.encode(chunked_data, convert_to_tensor=True)

        embeddings_list = embeddings.tolist()
        vectors = []
        for i, vector in enumerate(embeddings_list):
            vectors.append({"id": f'vector_{i}', "values": vector})
        self.index.upsert(vectors=vectors)

    def get_chain(self):
        embeddings = SentenceTransformerEmbeddings(model_name=os.environ.get("EMBEDDING_MODEL"))
        pc = LLMPinecone.from_existing_index(os.environ.get("PINECONE_INDEX_NAME"), embeddings)
        retriever = pc.as_retriever()
        self.chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
    def extract_criteria(self):
        try:
            response = self.chain.invoke("""
                - The response must be a json object and not a code type.
                - Select as many options as possible based on the provided location information, but the selected options must exist in predefined criteria's list.
            """)
            
            print(response)
            return response
        except Exception as e:
            print(f'Error extracting criteria with LangChain: {e}')
            return response['choices'][0]['text'].strip()
