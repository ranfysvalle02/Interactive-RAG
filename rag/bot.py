from typing import List
from actionweaver import RequireNext, action
from actionweaver.llms.azure.chat import ChatCompletion
from actionweaver.llms.openai.functions.tokens import TokenUsageTracker
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings import GPT4AllEmbeddings
from langchain.document_loaders import PlaywrightURLLoader
import openai
import serpapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
import params
import urllib.parse
from collections import Counter
openai.api_key = params.OPENAI_API_KEY
openai.api_version = params.OPENAI_API_VERSION
openai.api_type = params.OPENAI_TYPE #azure or openai

import pymongo 

MONGODB_URI = params.MONGODB_URI  
DATABASE_NAME = params.DATABASE_NAME
COLLECTION_NAME = params.COLLECTION_NAME

def get_unique_urls(collection):  
    urls = []  
    for item in collection:  
        # Extract the URL from the item in the collection  
        url = urllib.parse.urlparse(item['url']).netloc  
        urls.append(url)  
      
    unique_urls = set(urls)  
    url_counts = Counter(urls)  
      
    return unique_urls, url_counts  

class AzureAgent:
    def __init__(self, logger, st):
        self.rag_config = {
            "num_sources": 2,
            "source_chunk_size": 1000,
            "min_rel_score": 0,
            "unique": True
        }
        self.logger = logger
        self.text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        self.token_tracker = TokenUsageTracker(budget=None, logger=logger)
        if(params.OPENAI_TYPE != "azure"):
            from actionweaver.llms.openai.tools.chat import OpenAIChatCompletion
            self.llm = OpenAIChatCompletion(
                model="gpt-3.5-turbo",
                token_usage_tracker = TokenUsageTracker(budget=2000, logger=logger), 
                logger=logger)
        else:
            self.llm = ChatCompletion(
                model="gpt-3.5-turbo", 
                #model="gpt-4", 
                azure_deployment=params.OPENAI_AZURE_DEPLOYMENT,
                azure_endpoint=params.OPENAI_AZURE_ENDPOINT, api_key=params.OPENAI_API_KEY,
                api_version=params.OPENAI_API_VERSION, 
                token_usage_tracker = TokenUsageTracker(budget=2000, logger=logger), 
                logger=logger)
        self.messages = [
            {"role": "system", "content": "You are a resourceful AI assistant. You specialize in helping users build RAG pipelines interactively."},
            {"role": "system", "content": "Think critically and step by step. Do not answer directly."},
            {"role":"system", "content":"""\n\n[EXAMPLES]
            - User Input: What is MongoDB?
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "answer_question".
            - Action: "answer_question"('What is MongoDB?')

            - User Input: Reset chat history
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "reset_messages".
            - Action: "reset_messages"()

            - User Input: remove sources https://www.google.com, https://www.example.com
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "remove_source".
            - Action: "remove_source"(['https://www.google.com','https://www.example.com'])

            - User Input: read https://www.google.com, https://www.example.com
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "read_url".
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
    @action("iRAG", orch_expr=RequireNext(["iRAG"]), stop=True)
    def iRAG(self, num_sources:int, chunk_size: int):
        """
        Invoke this ONLY when the user asks you to change the RAG configuration.

        Parameters
        ----------
        num_sources : int
            how many documents should we use in the RAG pipeline?
        chunk_size : int
            how big should each chunk/source be?
        Returns successful response message. 

        -------
        str
            A message indicating success
        """
        with self.st.spinner(f"Changing RAG configuration..."):
            if num_sources > 0:
                self.rag_config["num_sources"] = int(num_sources)
            else:
                return f"Please provide a valid number of sources."
            if chunk_size > 0:
                self.rag_config["source_chunk_size"] = int(chunk_size)
            else:
                return f"Please provide a valid chunk size."
            print(self.rag_config)
            return f"New RAG config:{str(self.rag_config)}."
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
            {"role": "system", "content": "You are a resourceful AI assistant. You specialize in helping users build RAG pipelines interactively."},
            {"role": "system", "content": "Think critically and step by step. Do not answer directly."},
            {"role":"system", "content":"""\n\n[EXAMPLES]
            - User Input: What is MongoDB?
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "answer_question".
            - Action: "answer_question"('What is MongoDB?')

            - User Input: Reset chat history
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "reset_messages".
            - Action: "reset_messages"()

            - User Input: remove source https://www.google.com, https://www.example.com
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "remove_source".
            - Action: "remove_source"(['https://www.google.com','https://www.example.com'])

            - User Input: read https://www.google.com, https://www.example.com
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "read_url".
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
        with self.st.spinner(f"Searching '{query}'..."):
            search = serpapi.search({
                "q": str(query)+" -site:youtube.com",
                "location": "Austin, Texas, United States",
                "hl": "en",
                "gl": "us",
                "num":5,
                "google_domain": "google.com",
                "api_key": params.SERPAPI_KEY
            })
            res = search
            
            formatted_data = ""

            # Iterate through the data and append each item to the formatted_data string
            for idx, item in enumerate(res["organic_results"]):
                formatted_data += f"({idx}) {item['title']}: {item['snippet']}\n"
                formatted_data += f"[Source]: {item['link']}\n\n"

            return f"Here are the Google search results for '{query}':\n\n{formatted_data}\n"
   
    @action("remove_source", orch_expr=RequireNext(["remove_source"]), stop=True)
    def remove_source(self, urls: List[str]) -> str:
        """
        Invoke this if you need to remove one or more sources
        Args:
            urls (List[str]): The list of URLs to be removed
        Returns:
            str: Text with confirmation
        """
        with self.st.spinner(f"Deleting sources {', '.join(urls)}..."):
            self.collection.delete_many({"source": {"$in": urls}})
            return f"Sources ({', '.join(urls)}) successfully deleted.\n"


    def recall(self, text, n_docs=2, min_rel_score=0.25, chunk_max_length=800,unique=True):
        response = self.index.similarity_search_with_score(query=text, k=15)
        print(response)
        str_response = []
        tmp_docs = []
        print("recall"+"n_docs=="+str(n_docs))
        print("recall"+"min_rel_score=="+str(min_rel_score))
        for vs in response:
            score = vs[1]
            v = vs[0]
            print("URL"+v.metadata["source"]+";"+str(score))
            # lets only include stuff with at least X relevance
            if score < min_rel_score:
                continue
            else:
                if len(tmp_docs) == n_docs:
                    break
                if unique and v.metadata["source"] in tmp_docs:
                    continue
                str_response.append({"URL":v.metadata["source"],"content":v.page_content[:chunk_max_length]})
                tmp_docs.append(v.metadata["source"])
        
        if len(str_response)>0:
            return f"VectorStore Search Results[{len(str_response)}] (source=URL):\n{str_response}"[:5000]
        else:
            return "N/A"
    @action(name="get_sources_list", stop=True)
    def get_sources_list(self):
        """
        Invoke this to respond to list all the available sources in your knowledge base.
        Parameters
        ----------
        None
        """
        sources = self.collection.distinct("source")  
        
        if sources:  
            result = f"Available Sources [{len(sources)}]:\n"  
            result += "\n".join(sources[:5000])  
            return result  
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
        context_str = str(
            self.recall(
                query,
                n_docs=self.rag_config["num_sources"],
                min_rel_score=self.rag_config["min_rel_score"],
                chunk_max_length=self.rag_config["source_chunk_size"],
                unique=self.rag_config["unique"],
            )).strip()
        print("CTX"+context_str)
        if context_str == "N/A":
                return self.search_web(query)
        # if there is context available, let's try to answer the question
        PRECISE_SYS_PROMPT = """
        Given the following verified sources and a question, create a final concise answer in markdown. 
        If uncertain, search the web.

        Remember while answering:
            * The only verified sources are between START VERIFIED SOURCES and END VERIFIED SOURCES.
            * Only display images and links if they are found in the verified sources
            * If displaying images or links from the verified sources, copy the images and links exactly character for character and make sure the URL parameters are the same.
            * Only talk about the answer or reply with a follow up question, do not reference the verified sources.
            * Do not make up any part of an answer. 
            * If the answer isn't in or derivable from the verified sources search the web.
            * If the verified sources can answer the question in multiple different ways, ask a follow up question to clarify what the user wants to exactly to know about.
            * Questions might be vague or have multiple interpretations, you must ask follow up questions in this case.
            * You have access to the previous messages in the conversation which helps you help you answer questions that are related to previous questions. Always formulate your answer accounting for the previous messages.  
            * Final response must be less than 1200 characters.
            * Final response must begin with RAG config: __rag_config__

        [REQUIRED RESPONSE FORMAT]
        <concise well-formatted markdown response using the verified sources. include a sources section including the title, and the URL. Must be valid markdown.>

        [START VERIFIED SOURCES]
        __context_str__
        [END VERIFIED SOURCES]


        IMPORTANT! IF VERIFIED SOURCES DO NOT INCLUDE ENOUGH INFORMATION TO ANSWER, SEARCH THE WEB!

        [ACTUAL QUESTION. ANSWER BASED ON VERIFIED SOURCES]:
        __text__
        Begin! REMEMBER! IF VERIFIED SOURCES DO NOT INCLUDE ENOUGH INFORMATION TO ANSWER, SEARCH THE WEB!"""
        PRECISE_SYS_PROMPT = str(PRECISE_SYS_PROMPT).replace("__context_str__",context_str)
        PRECISE_SYS_PROMPT = str(PRECISE_SYS_PROMPT).replace("__text__",query)
        PRECISE_SYS_PROMPT = str(PRECISE_SYS_PROMPT).replace("__rag_config__",str(self.rag_config))

        print(PRECISE_SYS_PROMPT)
        self.messages += [{"role": "user", "content":PRECISE_SYS_PROMPT}]
        response = self.llm.create(messages=self.messages, actions = [
            self.search_web
        ], stream=True)
        return response
    def __call__(self, text):
        self.messages += [{"role": "user", "content":text}]
        response = self.llm.create(messages=self.messages, actions = [
            self.read_url,self.answer_question,self.remove_source,self.reset_messages,
            self.iRAG, self.get_sources_list
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
