import os
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as LLMPinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)


class AIModel:
    def __init__(self):
        index_name = os.environ.get("PINECONE_INDEX_NAME")
        self.txt_splitter = RecursiveCharacterTextSplitter(chunk_size=os.environ.get("TEXT_SPLITTER_CHUNK_SIZE"), chunk_overlap=os.environ.get("TEXT_SPLITTER_CHUNK_OVERLAP"))
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = SentenceTransformer(os.environ.get("EMBEDDING_MODEL"), device=device)
        self.embeddings = SentenceTransformerEmbeddings(model_name=os.environ.get("TEXT_SPLITTER_CHUNK_SIZE"))
        self.docsearch = Pinecone.from_existing_index(index_name, self.embeddings)

        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index = pc.Index(index_name)

        llm = ChatOpenAI(model_name=os.environ["OPENAI_API_MODEL"], openai_api_key=os.environ["OPENAI_API_KEY"])
        system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say 'I don't know'""")
        human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
        prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
        self.extracter = ConversationChain(prompt=prompt_template, llm=llm, verbose=True)

    def generate_embedding(self, info):
        _, yelp_feature, yelp_amenities, yelp_about, yelp_menu, yelp_name, yelp_reviews = info
        texts = [
            f'feature of {yelp_name} are {yelp_feature}',
            f'amenities of {yelp_name} are {yelp_amenities}',
            f"about of {yelp_name} are {yelp_about}",
            f'menu of {yelp_name} are {yelp_menu}',
            f'reviews of {yelp_name} are {yelp_reviews}',
        ]
        chunked_data = []
        for text in texts:
            chunked_data.extend(self.txt_splitter.split_text(text))

        embeddings = self.model.encode(chunked_data, convert_to_tensor=True)

        embeddings_list = embeddings.tolist()
        vectors = []
        for i, vector in enumerate(embeddings_list):
            vectors.append({"id": f'vector_{i}', "values": vector})
        return vectors

    def store_embedding(self, embedding):
        try:
            index_stats_response = self.index.describe_index_stats()
            print(index_stats_response)
            self.index.upsert(vectors=embedding)
            index_stats_response = self.index.describe_index_stats()
            print(index_stats_response)
            print('Embedding for location stored in Pinecone.')
        except Exception as e:
            print(f'Error storing embedding in Pinecone: {e}')
            raise e

    def extract_criteria(self, ):
        try:
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            context = find_match(refined_query)
            response = self.extracter.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            
            print(f'Criteria extracted for location: {response}')
            return response
        except Exception as e:
            print(f'Error extracting criteria with LangChain: {e}')
            return response['choices'][0]['text'].strip()
