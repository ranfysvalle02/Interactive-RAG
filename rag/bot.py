from typing import List
from actionweaver import RequireNext, action
from actionweaver.llms.azure.chat import ChatCompletion
from actionweaver.llms.openai.tools.chat import OpenAIChatCompletion
from actionweaver.llms.openai.functions.tokens import TokenUsageTracker
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings import GPT4AllEmbeddings
from langchain.document_loaders import PlaywrightURLLoader
from langchain.document_loaders import BraveSearchLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import params
import urllib.parse
import os
import pymongo
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
import utils

os.environ["OPENAI_API_KEY"] = params.OPENAI_API_KEY
os.environ["OPENAI_API_VERSION"] = params.OPENAI_API_VERSION
os.environ["OPENAI_API_TYPE"] = params.OPENAI_TYPE

MONGODB_URI = params.MONGODB_URI
DATABASE_NAME = params.DATABASE_NAME
COLLECTION_NAME = params.COLLECTION_NAME


class UserProxyAgent:
    def __init__(self, logger, st):
        self.rag_config = {
            "num_sources": 2,
            "source_chunk_size": 1000,
            "min_rel_score": 0.00,
            "unique": True,
        }
        self.init_messages = [
            {
                "role": "system",
                "content": "You are a resourceful AI assistant. You specialize in helping users build RAG pipelines interactively.",
            },
            {
                "role": "system",
                "content": "Think critically and step by step. Do not answer directly. Always take the most reasonable available action.",
            },
            {
                "role": "system",
                "content": "If user prompt is not related to modifying RAG strategy, resetting chat history, removing sources, learning sources, or a question - Respectfully decline to respond.",
            },
            {
                "role": "system",
                "content": """\n\n[EXAMPLES]
            - User Input: "What is kubernetes?"
            - Thought: I have an action available called "answer_question". I will use this action to answer the user's question about Kubernetes.
            - Observation: I have an action available called "answer_question". I will use this action to answer the user's question about Kubernetes.
            - Action: "answer_question"('What is kubernetes?')

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

            - User Input: add https://www.google.com, https://www.example.com
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "read_url".
            - Action: "read_url"(['https://www.google.com','https://www.example.com'])
            
            - User Input: learn https://www.google.com, https://www.example.com
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "read_url".
            - Action: "read_url"(['https://www.google.com','https://www.example.com'])
             
            - User Input: change chunk size to be 500 and num_sources to be 5
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "iRAG".
            - Action: "iRAG"(num_sources=5, chunk_size=500)
             
        [END EXAMPLES]\n\n
             
             ## IMPORTANT: 
                - DO NOT ANSWER DIRECTLY - ALWAYS USE AN ACTION/TOOL TO FORMULATE YOUR ANSWER
                - ALWAYS USE answer_question if USER PROMPT is a question
                - ALWAYS USE THE CORRECT TOOL/ACTION WHEN USER PROMPT IS related to modifying RAG strategy, resetting chat history, removing sources, learning sources
                - Always formulate your answer accounting for the previous messages
             
             REMEMBER! ALWAYS USE answer_question if USER PROMPT is a question
             """,
            },
        ]
        browser_options = Options()
        browser_options.headless = True
        browser_options.add_argument("--headless")
        browser_options.add_argument("--disable-gpu")
        self.browser = webdriver.Chrome(options=browser_options)

        self.logger = logger
        self.text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        self.token_tracker = TokenUsageTracker(budget=None, logger=logger)
        if params.OPENAI_TYPE != "azure":
            self.llm = OpenAIChatCompletion(
                model="gpt-3.5-turbo",
                token_usage_tracker=TokenUsageTracker(budget=2000, logger=logger),
                logger=logger,
            )
        else:
            self.llm = ChatCompletion(
                model="gpt-3.5-turbo",
                # model="gpt-4",
                azure_deployment=params.OPENAI_AZURE_DEPLOYMENT,
                azure_endpoint=params.OPENAI_AZURE_ENDPOINT,
                api_key=params.OPENAI_API_KEY,
                api_version=params.OPENAI_API_VERSION,
                token_usage_tracker=TokenUsageTracker(budget=2000, logger=logger),
                logger=logger,
            )
        self.messages = self.init_messages
        self.times = []
        self.gpt4all_embd = GPT4AllEmbeddings()
        self.client = pymongo.MongoClient(MONGODB_URI)
        self.db = self.client[DATABASE_NAME]
        self.collection = self.db[COLLECTION_NAME]
        self.vectorstore = MongoDBAtlasVectorSearch(self.collection, self.gpt4all_embd)
        self.index = self.vectorstore.from_documents(
            [], self.gpt4all_embd, collection=self.collection
        )
        self.st = st


class RAGAgent(UserProxyAgent):
    def preprocess_query(self, query):
        # Optional - Implement Pre-Processing for Security.
        # https://dev.to/jasny/protecting-against-prompt-injection-in-gpt-1gf8
        return query

    @action("iRAG", stop=True)
    def iRAG(
        self,
        num_sources: int,
        chunk_size: int,
        unique_sources: bool,
        min_rel_threshold: float,
    ):
        """
        Invoke this ONLY when the user explicitly asks you to change the RAG configuration in the most recent USER PROMPT.
        [EXAMPLE]
        - User Input: change chunk size to be 500 and num_sources to be 5
        
        Parameters
        ----------
        num_sources : int
            how many documents should we use in the RAG pipeline?
        chunk_size : int
            how big should each chunk/source be?
        unique_sources : bool
            include only unique sources? Y=True, N=False
        min_rel_threshold : float
            default=0.00; minimum relevance threshold to include a source in the RAG pipeline

        Returns successful response message.
        -------
        str
            A message indicating success
        """
        with self.st.spinner(f"Changing RAG configuration..."):
            if num_sources > 0:
                self.rag_config["num_sources"] = int(num_sources)
            else:
                self.rag_config["num_sources"] = 2
            if chunk_size > 0:
                self.rag_config["source_chunk_size"] = int(chunk_size)
            else:
                self.rag_config["source_chunk_size"] = 1000
            if unique_sources == True:
                self.rag_config["unique"] = True
            else:
                self.rag_config["unique"] = False
            if min_rel_threshold:
                self.rag_config["min_rel_score"] = min_rel_threshold
            else:
                self.rag_config["min_rel_score"] = 0.00
            print(self.rag_config)
            self.st.write(self.rag_config)
            return f"New RAG config:{str(self.rag_config)}."

    @action("read_url", stop=True)
    def read_url(self, urls: List[str]):
        """
        Invoke this ONLY when the user asks you to 'read', 'add' or 'learn' some URL(s).
        This function reads the content from specified sources, and ingests it into the Knowledgebase.
        URLs may be provided as a single string or as a list of strings.
        IMPORTANT! Use conversation history to make sure you are reading/learning/adding the right URLs.

        Parameters
        ----------
        urls : List[str]
            List of URLs to scrape.

        Returns
        -------
        str
            A message indicating successful reading of content from the provided URLs.
        """
        with self.st.spinner(f"```Analyzing the content in {urls}```"):
            loader = PlaywrightURLLoader(
                urls=urls, remove_selectors=["header", "footer"]
            )
            documents = loader.load_and_split(self.text_splitter)
            self.index.add_documents(documents)
            return f"```Contents in URLs {urls} have been successfully ingested (vector embeddings + content).```"

    @action("reset_messages", stop=True)
    def reset_messages(self) -> str:
        """
        Invoke this ONLY when the user asks you to reset chat history.

        Returns
        -------
        str
            A message indicating success
        """
        self.messages = self.init_messages
        self.st.empty()
        self.st.session_state.messages = []
        return f"Message history successfully reset."

    def encode_google_search(self, query):
        # Remove whitespace and replace with '+'
        query = query.strip().replace(" ", "+")
        # Encode the query using urllib.parse
        encoded_query = urllib.parse.quote(query)
        # Construct the Google search string
        search_string = f"https://www.google.com/search?q={encoded_query}&num=15"
        return search_string

    @action("search_web", stop=True)
    def search_web(self, query: str) -> List:
        """
        Invoke this if you need to search the web
        Args:
            query (str): The user's query
        Returns:
            str: Text with the Google Search results
        """
        with self.st.spinner(f"Searching '{query}'..."):
            # Use the headless browser to search the web
            self.browser.get(self.encode_google_search(query))
            html = self.browser.page_source
            soup = BeautifulSoup(html, "html.parser")
            search_results = soup.find_all("div", {"class": "g"})

            results = []
            links = []
            for i, result in enumerate(search_results):
                if result.find("h3") is not None:
                    if (
                        result.find("a")["href"] not in links
                        and "https://" in result.find("a")["href"]
                    ):
                        links.append(result.find("a")["href"])
                        results.append(
                            {
                                "title": utils.clean_text(result.find("h3").text),
                                "link": str(result.find("a")["href"]),
                            }
                        )

            df = pd.DataFrame(results)
            return f"Couldn't find enough information in my knowledge base. I need the right context from verified sources. \nTo improve the response: change the RAG strategy or add/remove sources. \nHere is what I found in the web for '{query}':\n{df.to_markdown()}\n\n"

    @action("remove_source", stop=True)
    def remove_source(self, urls: List[str]) -> str:
        """
        Invoke this if you need to remove one or more sources
        Args:
            urls (List[str]): The list of URLs to be removed
        Returns:
            str: Text with confirmation
        """
        with self.st.spinner(f"```Removing sources {', '.join(urls)}...```"):
            self.collection.delete_many({"source": {"$in": urls}})
            return f"```Sources ({', '.join(urls)}) successfully removed.```\n"

    def recall(
        self, text, n_docs=2, min_rel_score=0.25, chunk_max_length=800, unique=True
    ):
        # $vectorSearch
        print("recall=>" + str(text))
     
        try: 
            response = self.collection.aggregate(
                [
                    {
                        "$vectorSearch": {
                            "index": "default",
                            "queryVector": self.gpt4all_embd.embed_query(text),
                            "path": "embedding",
                            # "filter": {},
                            "limit": 15,  # Number (of type int only) of documents to return in the results. Value can't exceed the value of numCandidates.
                            "numCandidates": 50,  # Number of nearest neighbors to use during the search. You can't specify a number less than the number of documents to return (limit).
                        }
                    },
                    {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                    {"$match": {"score": {"$gte": min_rel_score}}},
                    {"$project": {"score": 1, "_id": 0, "source": 1, "text": 1}},
                ]
            )

        except pymongo.errors.OperationFailure as ex:  
            err_type = type(ex).__name__  
            err_args = ex.args  
            message = f"<b>Error! Please verify Atlas Search index exists.</b><hr/> An exception of type {err_type} occurred with the following arguments:\n{err_args}"  
            self.st.write(f"<div>{message}</div>", unsafe_allow_html=True)  
            raise  
        except Exception as ex:  
            err_type = type(ex).__name__  
            err_args = ex.args  
            message = f"<b>Error! An exception of type {err_type} occurred with the following arguments:\n{err_args}"  
            self.st.write("<div>{message}</div>", unsafe_allow_html=True)  
            raise  



        tmp_docs = []
        str_response = []
        for d in response:
            if len(tmp_docs) == n_docs:
                break
            if unique and d["source"] in tmp_docs:
                continue
            tmp_docs.append(d["source"])
            str_response.append(
                {
                    "URL": d["source"],
                    "content": d["text"][:chunk_max_length],
                    "score": d["score"],
                }
            )
        kb_output = (
            f"Knowledgebase Results[{len(tmp_docs)}]:\n```{str(str_response)}```\n## \n```SOURCES: "
            + str(tmp_docs)
            + "```\n\n"
        )
        self.st.write(kb_output)
        return str(kb_output)

    @action(name="get_sources_list", stop=True)
    def get_sources_list(self):
        """
        Invoke this to respond to list all the available sources in your knowledge base.
        Parameters
        ----------
        None
        """
        sources = self.collection.distinct("source")
        sources = [{"source": source} for source in sources]
        df = pd.DataFrame(sources)
        if sources:
            result = f"Available Sources [{len(sources)}]:\n"
            result += df.to_markdown()
            return result
        else:
            return "No sources found."

    @action(name="answer_question", stop=True)
    def answer_question(self, query: str):
        """
        ALWAYS TRY TO INVOKE THIS FIRST IF A USER ASKS A QUESTION.

        Parameters
        ----------
        query : str
            The query to be used for answering a question.
        """

        with self.st.spinner(f"Attemtping to answer question: {query}"):
            query = self.preprocess_query(query)
            context_str = str(
                self.recall(
                    query,
                    n_docs=self.rag_config["num_sources"],
                    min_rel_score=self.rag_config["min_rel_score"],
                    chunk_max_length=self.rag_config["source_chunk_size"],
                    unique=self.rag_config["unique"],
                )
            ).strip()
            PRECISE_PROMPT = """
            THINK CAREFULLY AND STEP BY STEP.
            WE WILL BE PLAYING A SPECIAL GAME. 

            Given the following verified sources and a question, using only the verified sources content create a final concise answer in markdown. 
            If VERIFIED SOURCES is not enough context to answer the question, THEN EXPLAIN YOURSELF AND KINDLY PERFORM A WEB SEARCH THE USERS BEHALF.

            Remember while answering:
                * The only verified sources are between START VERIFIED SOURCES and END VERIFIED SOURCES.
                * Only display images and links if they are found in the verified sources
                * If displaying images or links from the verified sources, copy the images and links exactly character for character and make sure the URL parameters are the same.
                * Do not make up any part of an answer. 
                * Questions might be vague or have multiple interpretations, you must ask follow up questions in this case.
                * Final response must be less than 1200 characters.
                * Final response must include total character count.
                * IF the verified sources can answer the question in multiple different ways, THEN respond with each of the possible answers.
                * Formulate your response using ONLY VERIFIED SOURCES. IF YOU CANNOT ANSWER THE QUESTION, THEN EXPLAIN YOURSELF AND KINDLY PERFORM A WEB SEARCH THE USERS BEHALF.

            [START VERIFIED SOURCES]
            __context_str__
            [END VERIFIED SOURCES]



            [ACTUAL QUESTION. ANSWER ONLY BASED ON VERIFIED SOURCES]:
            __text__

            # IMPORTANT! 
                * Final response must be expert quality markdown
                * The only verified sources are between START VERIFIED SOURCES and END VERIFIED SOURCES.
                * USE ONLY INFORMATION FROM VERIFIED SOURCES TO FORMULATE RESPONSE. IF VERIFIED SOURCES CANNOT ANSWER THE QUESTION, THEN EXPLAIN YOURSELF AND KINDLY PERFORM A WEB SEARCH THE USERS BEHALF.
                * Do not make up any part of an answer. 
            
            Begin!
            """
            PRECISE_PROMPT = str(PRECISE_PROMPT).replace("__context_str__", context_str)
            PRECISE_PROMPT = str(PRECISE_PROMPT).replace("__text__", query)

            print(PRECISE_PROMPT)
            SYS_PROMPT = """
                You are a helpful AI assistant. USING ONLY THE VERIFIED SOURCES, ANSWER TO THE BEST OF YOUR ABILITY.
                # IMPORTANT! 
                    * Final response must cite verified sources used in the answer (include URL).
                    * Final response must be expert quality markdown
                    * Must cite verified sources used in the answer (include URL) in a pretty format

                """
            # ReAct Prompt Technique
            EXAMPLE_PROMPT = """\n\n[EXAMPLES]
            - User Input: "What is kubernetes?"
            - Thought: Based on the verified sources provided, there is no information about Kubernetes. Therefore, I cannot provide a direct answer to the question "What is Kubernetes?" based on the verified sources. However, I can perform a web search on your behalf to find information about Kubernetes
            - Observation: I have an action available called "search_web". I will use this action to answer the user's question about Kubernetes.
            - Action: "search_web"('What is kubernetes?')

            - User Input: "What is MongoDB?"
            - Thought: Based on the verified sources provided, there is enough information about MongoDB. 
            - Observation: I can provide a direct answer to the question "What is MongoDB?" based on the verified sources.
            - Action: N/A
            [END EXAMPLES]

            [RESPONSE FORMAT]
            - Must be valid markdown
            - Must cite verified sources used in the answer (include URL) in a pretty format
            - Must be expert quality markdown. You are a technical writer with 30+ years of experience.
            """
            self.messages += [{"role": "user", "content": PRECISE_PROMPT}]
            response = self.llm.create(
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": PRECISE_PROMPT},
                    {"role": "system", "content": EXAMPLE_PROMPT},
                ],
                actions=[self.search_web],
                stream=False,
            )
            return response



    def __call__(self, text):
        text = self.preprocess_query(text)
        self.messages += [{"role": "user", "content": text}]
        if (
            len(self.messages) > 3
        ):  # just last three messages; history will usually be used for add/remove sources
            response = self.llm.create(
                messages=self.messages[-3:],
                actions=[
                    self.read_url,
                    self.answer_question,
                    self.remove_source,
                    self.reset_messages,
                    self.iRAG,
                    self.get_sources_list,
                    self.search_web,
                ],
                stream=False,
            )
        else:
            response = self.llm.create(
                messages=self.messages,
                actions=[
                    self.read_url,
                    self.answer_question,
                    self.remove_source,
                    self.reset_messages,
                    self.iRAG,
                    self.get_sources_list,
                    self.search_web,
                ],
                stream=False,
            )
        return response

    def print_output(output):
        from collections.abc import Iterable

        if isinstance(output, str):
            print(output)
        elif isinstance(output, Iterable):
            for chunk in output:
                content = chunk.choices[0].delta.content
                if content is not None:
                    print(content, end="")

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
