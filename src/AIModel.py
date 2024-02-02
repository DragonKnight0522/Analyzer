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

        Question: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(template)
        # self.prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)

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
            self.index.upsert(vectors=embedding)
            # index_stats_response = self.index.describe_index_stats()
            # print(index_stats_response)
            print('Embedding for location stored in Pinecone.')
        except Exception as e:
            print(f'Error storing embedding in Pinecone: {e}')
            raise e

    def get_chain(self):
        embeddings = SentenceTransformerEmbeddings(model_name=os.environ.get("EMBEDDING_MODEL"))
        pc = LLMPinecone.from_existing_index(os.environ.get("PINECONE_INDEX_NAME"), embeddings)
        retriever = pc.as_retriever()
        # self.chain = ConversationalRetrievalChain.from_llm(self.llm, retriever=retriever, verbose=True)
        # self.chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=self.prompt_template)
        self.chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
    def extract_criteria(self):
        try:
            # response = self.chain.run({"context": "extract data", "chat_history" : "", "question": "give me bar type of this location"})
            response = self.chain.invoke("What is the name of the location?")
            
            print(f'Criteria extracted for location: {response}')
            return response
        except Exception as e:
            print(f'Error extracting criteria with LangChain: {e}')
            return response['choices'][0]['text'].strip()
