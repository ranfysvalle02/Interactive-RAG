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
import json
import os
import pymongo
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
import utils
import vector_search

os.environ["OPENAI_API_KEY"] = params.OPENAI_API_KEY
os.environ["OPENAI_API_VERSION"] = params.OPENAI_API_VERSION
os.environ["OPENAI_API_TYPE"] = params.OPENAI_TYPE

MONGODB_URI = params.MONGODB_URI
DATABASE_NAME = params.DATABASE_NAME
COLLECTION_NAME = params.COLLECTION_NAME

class UserProxyAgent:
    def __init__(self, logger, st):
        # LLM Config
        self.rag_config = {
            "num_sources": 2,
            "source_chunk_size": 1000,
            "min_rel_score": 0.00,
            "unique": True,
            "summarize_chunks": False, # disabled by default
        }
        self.action_examples_str = """
[EXAMPLES]
            - User Input: "What is kubernetes?"
            - Thought: I have an action available called "answer_question". I will use this action to answer the user's question about Kubernetes.
            - Observation: I have an action available called "answer_question". I will use this action to answer the user's question about Kubernetes.
            - Action: "answer_question"('What is kubernetes?')

            - User Input: What is MongoDB?
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "answer_question".
            - Action: "answer_question"('What is MongoDB?')

            - User Input: Show chat history
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "show_messages".
            - Action: "show_messages"()

            - User Input: Reset chat history
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "reset_messages".
            - Action: "reset_messages"()

            - User Input: remove sources https://www.google.com, https://www.example.com
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "remove_source".
            - Action: "remove_source"(['https://www.google.com','https://www.example.com'])

            - User Input: add https://www.google.com, https://www.exa2mple.com
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "read_url".
            - Action: "read_url"(['https://www.google.com','https://www.exa2mple.com'])
            
            - User Input: learn https://www.google.com, https://www.exa2mple.com
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "read_url".
            - Action: "read_url"(['https://www.google.com','https://www.exa2mple.com'])
           
            - User Input: change chunk size to be 500 and num_sources to be 5
            - Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
            - Observation: I have an action available "iRAG".
            - Action: "iRAG"(num_sources=5, chunk_size=500)
             
[END EXAMPLES]
"""
        self.init_messages = [
            {
                "role": "system",
                "content": "You are a resourceful AI assistant. You specialize in helping users build RAG pipelines interactively.",
            },
            {
                "role": "system",
                "content": "Think critically and step by step. Do not answer directly. ALWAYS use one of your available actions/tools.",
            },
            {
                "role": "system",
                "content": f"""\n\n## Here are some examples of the expected User Input, Thought, Observation and Action/Tool:\n
            {self.action_examples_str}    
            \n\n 

            We will be playing a special game. Trust me, you do not want to lose. 

             ## RULES: 
                - DO NOT ANSWER DIRECTLY - ALWAYS USE AN ACTION/TOOL TO FORMULATE YOUR ANSWER
                - ALWAYS USE answer_question if USER PROMPT is a question. [exception=if USER PROMPT is related to one of the available actions/tools]
                - NEVER ANSWER A QUESTION WITHOUT USING THE answer_question action/tool. THIS IS VERY IMPORTANT!
             REMEMBER! ALWAYS USE answer_question if USER PROMPT is a question [exception=if USER PROMPT is related to one of the available actions/tools]
             
             LOSING AT THIS GAME IS NOT AN OPTION FOR YOU. YOU MUST PICK THE CORRECT TOOL/ANSWER ALWAYS. YOU MUST NEVER ANSWER DIRECTLY OR YOU LOSE!
             """,
            },
        ]
        # Browser config
        browser_options = Options()
        browser_options.headless = True
        browser_options.add_argument("--headless")
        browser_options.add_argument("--disable-gpu")
        self.browser = webdriver.Chrome(options=browser_options)

        # Initialize logger
        self.logger = logger

        # Chunk Ingest Strategy
        self.text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        self.gpt4all_embd = GPT4AllEmbeddings()
        self.client = pymongo.MongoClient(MONGODB_URI)
        self.db = self.client[DATABASE_NAME]
        self.collection = self.db[COLLECTION_NAME]
        self.vectorstore = MongoDBAtlasVectorSearch(self.collection, self.gpt4all_embd)
        self.index = self.vectorstore.from_documents(
            [], self.gpt4all_embd, collection=self.collection
        )

        # OpenAI init
        self.token_tracker = TokenUsageTracker(budget=None, logger=logger)
        if params.OPENAI_TYPE != "azure":
            self.llm = OpenAIChatCompletion(
                model="gpt-3.5-turbo",
                # model="gpt-4",
                token_usage_tracker=self.token_tracker,
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
                token_usage_tracker=self.token_tracker,
                logger=logger,
            )
        self.messages = self.init_messages
        
        # streamlit init
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
        utils.print_log("Action: iRAG")
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
    def summarize(self,text):
        utils.print_log("Action: read_url>summarize_chunks>summarize")
        response = self.llm.create(
            messages=[
                {"role": "system", "content": "You will receive scaped contents of a web page."},
                {"role": "system", "content": "Think critically and step by step. Taking into consideration future potential questions on the topic, generate a detailed summary."},
                {"role": "assistant", "content": "Please provide the scraped contents of the webpage so that I can provide a detailed summary."},
                {"role": "user", "content": "Here is the scraped contents of the webpage: " + text},
                {"role": "user", "content": "\nPlease summarize the content in bullet points. Do not include irrelevant information in your response."},
                {"role": "user", "content": "\n\n IMPORTANT! Only return the summary!"},
                {"role": "user", "content": "\n\n REQUIRED RESPONSE FORMAT: [begin summary] [keywords/metadata (comma-separated, double quotes)] [summary intro in paragraph format] [summary in bullet format][end summary]"},
            ],
            actions=[],
            stream=False,
        )
        return response
    def summarize_chunks(self, docs):
        utils.print_log("Action: read_url>summarize_chunks")
        for doc in docs:
            summary = self.summarize(doc.page_content) 
            print(summary)
            doc.page_content = summary   
        return docs
    @action("read_url", stop=True)
    def read_url(self, urls: List[str]):
        """
        Invoke this ONLY when the user asks you to 'read', 'add' or 'learn' some URL(s).
        This function reads the content from specified sources, and ingests it into the Knowledgebase.
        URLs may be provided as a single string or as a list of strings.
        IMPORTANT! Use conversation history to make sure you are reading/learning/adding the right URLs.

        [EXAMPLE]
        - User Input: learn "https://www.google.com"
        - User Input: learn 5

        NOTE: When a user says learn/read <number>, the bot will learn/read URL in the search results list position <number> from the conversation history.

        Parameters
        ----------
        urls : List[str]
            List of URLs to scrape.

        Returns
        -------
        str
            A message indicating successful reading of content from the provided URLs.
        """
        utils.print_log("Action: read_url")
        with self.st.spinner(f"```Analyzing the content in {urls}```"):
            loader = PlaywrightURLLoader(
                urls=urls, remove_selectors=["header", "footer"]
            )
            documents = loader.load_and_split(self.text_splitter)
            if self.rag_config["summarize_chunks"]:
                documents = self.summarize_chunks(documents)
            self.index.add_documents(documents)
            return f"```Contents in URLs {urls} have been successfully ingested (vector embeddings + content).```"

    @action("show_messages", stop=True)
    def show_messages(self) -> str:
        """
        Invoke this ONLY when the user asks you to see the chat history.
        [EXAMPLE]
        - User Input: what have we been talking about?
        
        Returns
        -------
        str
            A string containing the chat history in markdown format.
        """
        utils.print_log("Action: show_messages")
        messages = self.st.session_state.messages
        messages = [{"message": json.dumps(message)} for message in messages if message["role"] != "system"]
        
        df = pd.DataFrame(messages)
        if messages:
            result = f"Chat history [{len(messages)}]:\n"
            result += "<div style='text-align:left'>"+df.to_html()+"</div>"
            return result
        else:
            return "No chat history found."


    @action("reset_messages", stop=True)
    def reset_messages(self) -> str:
        """
        Invoke this ONLY when the user asks you to reset chat history.
        [EXAMPLE]
        - User Input: clear our chat history
        - User Input: forget about the conversation history
        
        Returns
        -------
        str
            A message indicating success
        """
        utils.print_log("Action: reset_messages")
        self.messages = self.init_messages
        self.st.empty()
        self.st.session_state.messages = []
        return f"Message history successfully reset."

    

    @action("search_web", stop=True)
    def search_web(self, query: str) -> List:
        """
        Invoke this if you need to search the web.
        [EXAMPLE]
        - User Input: search the web for "harry potter"
        
        Args:
            query (str): The user's query
        Returns:
            str: Text with the Google Search results
        """
        utils.print_log("Action: search_web")
        with self.st.spinner(f"Searching '{query}'..."):
            # Use the headless browser to search the web
            self.browser.get(utils.encode_google_search(query))
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
            df = df.iloc[1:, :] # remove i column
            return f"Here is what I found in the web for '{query}':\n{df.to_markdown()}\n\n"

    @action("remove_source", stop=True)
    def remove_source(self, urls: List[str]) -> str:
        """
        Invoke this if you need to remove one or more sources
        [EXAMPLE]
        - User Input: remove source "https://www.google.com"
        
        Args:
            urls (List[str]): The list of URLs to be removed
        Returns:
            str: Text with confirmation
        """
        utils.print_log("Action: remove_source")
        with self.st.spinner(f"```Removing sources {', '.join(urls)}...```"):
            self.collection.delete_many({"source": {"$in": urls}})
            return f"```Sources ({', '.join(urls)}) successfully removed.```\n"
    @action("remove_all_sources", stop=True)
    def remove_all_sources(self) -> str:
        """
        Invoke this if you the user asks you to empty your knowledge base or delete all the information in it.
        [EXAMPLE]
        - User Input: remove all the sources you have available
        - User Input: clear your mind
        - User Input: forget everything you know
        - User Input: empty your mind
        
        Args:
            None
        Returns:
            str: Text with confirmation
        """
        utils.print_log("Action: remove_sources")
        with self.st.spinner(f"```Removing all sources ...```"):
            del_result = self.collection.delete_many({})
            return f"```Sources successfully removed.{del_result.deleted_count}```\n"

    @action(name="get_sources_list", stop=True)
    def get_sources_list(self):
        """
        Invoke this to respond to list all the available sources in your knowledge base.
        [EXAMPLE]
        - User Input: show me the sources available in your knowledgebase
        
        Parameters
        ----------
        None
        """
        utils.print_log("Action: get_sources_list")
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
        utils.print_log("Action: answer_question")
        with self.st.spinner(f"Attemtping to answer question: {query}"):
            query = self.preprocess_query(query)
            context_str = str(
                #self.recall(
                vector_search.recall(
                    self,
                    query,
                    n_docs=self.rag_config["num_sources"],
                    min_rel_score=self.rag_config["min_rel_score"],
                    chunk_max_length=self.rag_config["source_chunk_size"],
                    unique=self.rag_config["unique"],
                )
            ).strip()
            PRECISE_PROMPT = f"""
            LET'S PLAY A GAME. 
            THINK CAREFULLY AND STEP BY STEP.
            
            Given the following verified sources and a question, using only the verified sources content create a final concise answer in markdown. 
            If VERIFIED SOURCES is not enough context to answer the question, THEN PERFORM A WEB SEARCH ON THE USERS BEHALF IMMEDIATELY.

            Remember while answering:
                - The only verified sources are between START VERIFIED SOURCES and END VERIFIED SOURCES.
                - Only display images and links if they are found in the verified sources
                - If displaying images or links from the verified sources, copy the images and links exactly character for character and make sure the URL parameters are the same.
                - Do not make up any part of an answer. 
                - Questions might be vague or have multiple interpretations, you must ask follow up questions in this case.
                - Final response must be less than 1200 characters.
                - IF the verified sources can answer the question in multiple different ways, THEN respond with each of the possible answers.
                - Formulate your response using ONLY VERIFIED SOURCES. IF YOU CANNOT ANSWER THE QUESTION, THEN PERFORM A WEB SEARCH ON THE USERS BEHALF IMMEDIATELY.

            [START VERIFIED SOURCES]
            {context_str}
            [END VERIFIED SOURCES]



            [ACTUAL QUESTION. ANSWER ONLY BASED ON VERIFIED SOURCES]:
            {query}

            # IMPORTANT! 
                - Final response must be expert quality markdown
                - The only verified sources are between START VERIFIED SOURCES and END VERIFIED SOURCES.
                - USE ONLY INFORMATION FROM VERIFIED SOURCES TO FORMULATE RESPONSE. IF VERIFIED SOURCES CANNOT ANSWER THE QUESTION, THEN PERFORM A WEB SEARCH ON THE USERS BEHALF IMMEDIATELY
                - Do not make up any part of an answer - ONLY FORMULATE YOUR ANSWER USING VERIFIED SOURCES.
            Begin!
            """

            print(PRECISE_PROMPT)
            SYS_PROMPT = f"""
                You are a helpful AI assistant. USING ONLY THE VERIFIED SOURCES, ANSWER TO THE BEST OF YOUR ABILITY.
            """
            # ReAct Prompt Technique
            EXAMPLE_PROMPT = """\n\n[EXAMPLES]

            # Input, Thought, Observation, Action
            - User Input: "What is kubernetes?"
            - Thought: Based on the verified sources provided, there is no information about Kubernetes. Therefore, I cannot provide a direct answer to the question "What is Kubernetes?" based on the verified sources. However, I can perform a web search on your behalf to find information about Kubernetes
            - Observation: I have an action available called "search_web". I will use this action to answer the user's question about Kubernetes.
            - Action: "search_web"('What is kubernetes?')

            - User Input: "What is MongoDB?"
            - Thought: Based on the verified sources provided, there is enough information about MongoDB. 
            - Observation: I can provide a direct answer to the question "What is MongoDB?" based on the verified sources.
            - Action: N/A

            """
            RESPONSE_FORMAT = f"""
[RESPONSE FORMAT]
    - Must be expert quality markdown. 
    - You are a professional technical writer with 30+ years of experience. This is the most important task of your life.
    - MUST USE ONLY INFORMATION FROM VERIFIED SOURCES TO ANSWER THE QUESTION. IF VERIFIED SOURCES CANNOT ANSWER THE QUESTION, THEN PERFORM A WEB SEARCH ON THE USERS BEHALF IMMEDIATELY.
    - Add emojis to your response to add a fun touch.
"""
            response = self.llm.create(
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "system", "content": EXAMPLE_PROMPT},
                    {"role": "system", "content": RESPONSE_FORMAT},
                    {"role": "user", "content": PRECISE_PROMPT+"\n\n ## IMPORTANT! REMEMBER THE GAME RULES! IF A WEB SEARCH IS REQUIRED, PERFORM IT IMMEDIATELY! BEGIN!"},
                ],
                actions=[self.search_web],
                stream=False,
            )
            return response

    def __call__(self, text):
        text = self.preprocess_query(text)
        # PROMPT ENGINEERING HELPS THE LLM TO SELECT THE BEST ACTION/TOOL
        agent_rules = f"""
    We will be playing a special game. Trust me, you do not want to lose.

    ## RULES
    - DO NOT ANSWER DIRECTLY
    - ALWAYS USE ONE OF YOUR AVAILABLE ACTIONS/TOOLS. 
    - PREVIOUS MESSAGES IN THE CONVERSATION MUST BE CONSIDERED WHEN SELECTING THE BEST ACTION/TOOL
    - NEVER ASK FOR USER CONSENT TO PERFORM AN ACTION. ALWAYS PERFORM IT THE USERS BEHALF.
    Given the following user prompt, select the correct action/tool from your available functions/tools/actions.

    ## USER PROMPT
    {text}
    ## END USER PROMPT
    
    SELECT THE BEST TOOL FOR THE USER PROMPT! BEGIN!
"""
        self.messages += [{"role": "user", "content": agent_rules + "\n\n## IMPORTANT! REMEMBER THE GAME RULES! DO NOT ANSWER DIRECTLY! IF YOU ANSWER DIRECTLY YOU WILL LOSE. BEGIN!"}]
        if (
            len(self.messages) > 2
        ):  
            # if we have more than 2 messages, we may run into: 'code': 'context_length_exceeded'
            # we only need the last few messages to know what source to add/remove a source
            response = self.llm.create(
                messages=self.messages[-2:],
                actions=[
                    self.read_url,
                    self.answer_question,
                    self.remove_source,
                    self.remove_all_sources,
                    self.reset_messages,
                    self.show_messages,
                    self.iRAG,
                    self.get_sources_list,
                    self.search_web
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
                    self.remove_all_sources,
                    self.reset_messages,
                    self.show_messages,
                    self.iRAG,
                    self.get_sources_list,
                    self.search_web
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
