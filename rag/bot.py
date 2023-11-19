from typing import List
from actionweaver import RequireNext, action
from actionweaver.llms.azure.chat import ChatCompletion
from actionweaver.llms.openai.tokens import TokenUsageTracker
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings import GPT4AllEmbeddings
from langchain.document_loaders import PlaywrightURLLoader
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter



openai.api_key = ""
openai.api_version = "2023-10-01-preview"
openai.api_type = "azure"

import pymongo 

MONGODB_URI = ""  
DATABASE_NAME = ""  
COLLECTION_NAME = ""


class AzureAgent:
    def __init__(self, logger, st):
        self.logger = logger
        self.text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        self.token_tracker = TokenUsageTracker(budget=None, logger=logger)
        self.llm = ChatCompletion(
            model="gpt-3.5-turbo", azure_deployment="",
            azure_endpoint="https://.openai.azure.com/", api_key="",
            api_version="2023-10-01-preview", 
            token_usage_tracker = TokenUsageTracker(budget=2000, logger=logger), 
            logger=logger)

        self.messages = [
            {"role": "system", "content": "You are a resourceful AI assistant.."},
            {"role": "system", "content": "Think critically and step by step. Do not answer directly."},
            {"role":"system", "content":"""\n\n[EXAMPLES]
            - User Input: What is MongoDB?
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "answer_question". I'll use that
            - Action: "answer_question"('What is MongoDB?')

            - User Input: Reset chat history
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "reset_messages". I'll use that
            - Action: "reset_messages"()

            - User Input: remove source https://www.google.com
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "remove_source". I'll use that
            - Action: "remove_source"('https://www.google.com')

            - User Input: read https://www.google.com, https://www.example.com
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "read_url". I'll use that
            - Action: "read_url"(['https://www.google.com','https://www.example.com'])
        [END EXAMPLES]\n\n"""}
                         ]
        self.times = []
        self.gpt4all_embd = GPT4AllEmbeddings()
        self.client = pymongo.MongoClient(MONGODB_URI)
        self.db = self.client[DATABASE_NAME]  
        self.collection = self.db[COLLECTION_NAME]  
        self.vectorstore = MongoDBAtlasVectorSearch(self.collection, self.gpt4all_embd)  
        self.index = self.vectorstore.from_documents([], self.gpt4all_embd, collection=self.collection)
        self.st = st



class RAGAgent(AzureAgent):
    @action("read_url", orch_expr=RequireNext(["read_url"]), stop=True)
    def read_url(self, urls: List[str]):
        """
        Invoke this ONLY when the user asks you to read. This function reads the content from specified sources.

        Parameters
        ----------
        urls : List[str]
            List of URLs to scrape.

        Returns
        -------
        str
            A message indicating successful reading of content from the provided URLs.
        """
        with self.st.spinner(f"Learning the content in {urls}"):
            loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])  
            documents = loader.load_and_split(self.text_splitter)
            self.index.add_documents(
                    documents
            )       
            return f"Contents in URLs {urls} have been successfully learned."
    @action("reset_messages", orch_expr=RequireNext(["reset_messages"]), stop=True)
    def reset_messages(self) -> str:
        """
        Invoke this ONLY when the user asks you to reset chat history.

        Returns
        -------
        str
            A message indicating success
        """
        self.messages = [
            {"role": "system", "content": "You are a resourceful AI assistant.."},
            {"role": "system", "content": "Think critically and step by step. Do not answer directly."},
            {"role":"system", "content":"""\n\n[EXAMPLES]
            - User Input: What is MongoDB?
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "answer_question". I'll use that
            - Action: "answer_question"('What is MongoDB?')

            - User Input: Reset chat history
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "reset_messages". I'll use that
            - Action: "reset_messages"()

            - User Input: remove source https://www.google.com
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "remove_source". I'll use that
            - Action: "remove_source"('https://www.google.com')

            - User Input: read https://www.google.com, https://www.example.com
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "read_url". I'll use that
            - Action: "read_url"(['https://www.google.com','https://www.example.com'])
        [END EXAMPLES]\n\n"""}
                         ]
        return f"Message history successfully reset."
    
    @action("search_web", orch_expr=RequireNext(["search_web"]), stop=True)
    def search_web(self,query:str) -> List:
        """
        Invoke this if you need to search the web

        Args:
            query (str): The user's query

        Returns:
            str: Text with the Google Search results
        """
        from serpapi import GoogleSearch
        with self.st.spinner(f"Searching '{query}'..."):
            search = GoogleSearch({
                "q": str(query)+" -site:youtube.com",
                "location": "Austin, Texas, United States",
                "hl": "en",
                "gl": "us",
                "num":5,
                "google_domain": "google.com",
                "api_key": ""
            })
            res = search.get_dict()
            
            formatted_data = ""

            # Iterate through the data and append each item to the formatted_data string
            for idx, item in enumerate(res["organic_results"]):
                formatted_data += f"({idx}) {item['title']}: {item['snippet']}\n"
                formatted_data += f"[Source]: {item['link']}\n\n"

            return f"Here are the Google search results for '{query}':\n\n{formatted_data}\n"
    @action("remove_source", orch_expr=RequireNext(["remove_source"]), stop=True)
    def remove_source(self,url:str) -> List:
        """
        Invoke this if you need to remove a source. User query will begin with: "remove source {url_goes_here}"

        Args:
            url (str): The url to be removed

        Returns:
            str: Text with confirmation
        """
        with self.st.spinner(f"Deleting source '{url}'..."):
            self.collection.delete_many({"source":str(url)})
            return f"Source ("+str(url)+") successfully deleted.\n"

    def recall(self, text):
        response = self.index.similarity_search_with_score(text) #default to 2 chunks for simplicity
        str_response = []
        for vs in response:
            score = vs[1]
            v = vs[0]
            print("URL"+v.metadata["source"]+";"+str(score))
            str_response.append({"URL":v.metadata["source"],"content":v.page_content[:800]})
        
        if len(str_response)>0:
            return f"VectorStore Search Results (source=URL):\n{str_response}"[:5000]
        else:
            return "N/A"
    @action(name="answer_question", stop=True)
    def answer_question(self, query: str):
        """
        Invoke this to respond to a question.
        Parameters
        ----------
        query : str
            The query to be used for answering a question.
        """
        context_str = str(self.recall(query)).strip()
        print("CTX"+context_str)
        if context_str == "N/A":
                return self.search_web(query)
        PRECISE_SYS_PROMPT = """
        Given the following verified sources and a question, create a final concise answer in markdown. If uncertain, search the web.

        Remember while answering:
            * The only verified sources are between START VERIFIED SOURCES and END VERIFIED SOURCES.
            * Only display images and links if they are found in the verified sources
            * If displaying images or links from the verified sources, copy the images and links exactly character for character and make sure the URL parameters are the same.
            * Only talk about the answer or reply with a follow up question, do not reference the verified sources.
            * Do not make up any part of an answer. If the answer isn't in or derivable from the verified sources search the web.
            * If the verified sources can answer the question in multiple different ways, ask a follow up question to clarify what the user wants to exactly to know about.
            * Questions might be vague or have multiple interpretations, you must ask follow up questions in this case.
            * You have access to the previous messages in the conversation which helps you help you answer questions that are related to previous questions. Always formulate your answer accounting for the previous messages.  
            * Final response must include source URLs for reference in the Footnotes section.
            * Final response must be less than 1000 characters.
        [START VERIFIED SOURCES]
        __context_str__
        [END VERIFIED SOURCES]

        [ACTUAL QUESTION BASED ON VERIFIED SOURCES]:
        __text__
        Begin!"""
        PRECISE_SYS_PROMPT = str(PRECISE_SYS_PROMPT).replace("__context_str__",context_str)
        PRECISE_SYS_PROMPT = str(PRECISE_SYS_PROMPT).replace("__text__",query)
        print(PRECISE_SYS_PROMPT)
        self.messages += [{"role": "user", "content":PRECISE_SYS_PROMPT}]
        response = self.llm.create(messages=self.messages, actions = [
            self.search_web
        ], stream=True)
        return response
    def __call__(self, text):
        self.messages += [{"role": "user", "content":text}]
        response = self.llm.create(messages=self.messages, actions = [
            self.read_url,self.answer_question,self.remove_source,self.reset_messages
        ], stream=True)

        return response



def print_output(output):
    from collections.abc import Iterable
    if isinstance(output, str):
        print (output)
    elif isinstance(output, Iterable):
        for chunk in output:
            content = chunk.choices[0].delta.content
            if content is not None:
                print(content, end='')


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        filename="bot.log",
        filemode="a",
        format="%(asctime)s.%(msecs)04d %(levelname)s {%(module)s} [%(funcName)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    agent = RAGAgent(logger, None)
