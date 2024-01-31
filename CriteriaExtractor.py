import openai
from langchain.llms import OpenAI

class CriteriaExtractor:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.llm = None

    def initialize_chain(self):
        openai.api_key = self.openai_api_key
        self.llm = OpenAI()

    def extract_criteria(self, prompt):
        try:
            response = self.llm.complete(prompt=prompt)
            extracted_text = response['choices'][0]['text'].strip()
            print(f'Criteria extracted for location: {extracted_text}')
            return extracted_text
        except Exception as e:
            print(f'Error extracting criteria with LangChain: {e}')
            return response['choices'][0]['text'].strip()
