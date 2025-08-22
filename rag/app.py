import os  
import logging  
import urllib.parse  
import re  
import json  
import inspect  
import time  
import uuid  
from typing import List, Dict, Any, Optional  
import traceback  
import tempfile  
import io  
import concurrent.futures  
  
# --- 0. DEPENDENCY MANAGEMENT ---  
# pip install pymongo pandas beautifulsoup4 flask openai python-dotenv flask-cors requests langchain  
# pip install langchain-openai langchain-mongodb ddgs docling langchain-voyageai voyageai  
# IMPORTANT: If you had "duckduckgo-search", remove it: pip uninstall duckduckgo-search -y  
# Then install ddgs: pip install ddgs  
  
# --- Core Libraries ---  
import pymongo  
from flask import Flask, request, jsonify, render_template  
from flask_cors import CORS  
from pymongo.operations import SearchIndexModel  
from pymongo.errors import OperationFailure, ConnectionFailure  
from dotenv import load_dotenv  
import requests  
from bson.objectid import ObjectId  
from ddgs import DDGS  
from docling.document_converter import DocumentConverter  
  
# --- LangChain Imports ---  
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings  
from langchain_voyageai import VoyageAIEmbeddings  
from langchain_mongodb import MongoDBAtlasVectorSearch  
from langchain.agents import AgentExecutor, create_openai_tools_agent  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from langchain_core.messages import HumanMessage, AIMessage  
from langchain_core.tools import tool  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
  
# --- 1. SET ENVIRONMENT VARIABLES & CONFIGURATION ---  
load_dotenv()  
  
DATABASE_NAME = "interactive_rag_db"  
COLLECTION_NAME = "knowledge_base_sessions"  
SESSION_FIELD = "session_id"  
  
# Embedding setup for multiple models  
EMBEDDING_CONFIG = {  
    "openai": {  
        "vector_field": "embedding_openai",  
        "index_name": "openai_vector_index",  
        "dimensions": 1536  
    },  
    "voyageai": {  
        "vector_field": "embedding_voyageai",  
        "index_name": "voyageai_vector_index",  
        "dimensions": 1024  
    }  
}  
  
# --- 2. UTILITY & LOGGING ---  
logging.basicConfig(  
    filename="rag_agent.log",  
    filemode="a",  
    format="%(asctime)s - %(levelname)s - %(message)s",  
    level=logging.INFO  
)  
logger = logging.getLogger(__name__)  
  
def print_log(message: str):  
    print(message)  
    logger.info(message)  
  
# --- 3. AGENT STATE & CONFIG ---  
class AgentConfig:  
    def __init__(self):  
        self.rag_config = {  
            "num_sources": 3,  
            "min_rel_score": 0.0,  
            "max_chunk_length": 2000  
        }  
        self.embedding_clients = {}  
        try:  
            self.db_client = pymongo.MongoClient(  
                os.getenv("MDB_URI"),  
                serverSelectionTimeoutMS=10000  
            )  
            self.db_client.admin.command('ping')  
            print_log("[INFO] MongoDB connection successful.")  
        except (ConnectionFailure, OperationFailure) as e:  
            print_log(f"[FATAL] 🚨 MongoDB connection failed. Error: {e}")  
            raise  
  
        self.db = self.db_client[DATABASE_NAME]  
        self.collection = self.db[COLLECTION_NAME]  
  
        self.chat_history = {}  
        self.current_session = "default"  
        self.last_retrieved_sources = []  
  
        print_log("--- 🧠 Initializing Embedding Clients ---")  
        # OpenAI Embeddings  
        self.embedding_clients["openai"] = AzureOpenAIEmbeddings(  
            azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002"),  
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
            api_key=os.getenv("AZURE_OPENAI_API_KEY")  
        )  
        print_log("[INFO] OpenAI embedding client initialized.")  
  
        # VoyageAI Embeddings (optional)  
        if os.getenv("VOYAGE_API_KEY"):  
            try:  
                self.embedding_clients["voyageai"] = VoyageAIEmbeddings(  
                    model="voyage-2",  
                    voyage_api_key=os.getenv("VOYAGE_API_KEY")  
                )  
                print_log("[INFO] VoyageAI embedding client initialized.")  
            except Exception as e:  
                print_log(f"[WARN] ⚠️ VoyageAI initialization failed: {e}. Skipping.")  
        else:  
            print_log("[INFO] VOYAGE_API_KEY not found. VoyageAI embeddings not available.")  
  
        print_log("------------------------------------")  
  
config = AgentConfig()  
  
# --- 4. BACKGROUND TASK SETUP ---  
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)  
tasks = {}  
  
def run_ingestion_task(  
    task_id: str,  
    content: str,  
    source: str,  
    source_type: str,  
    session_id: str,  
    chunk_size: int,  
    chunk_overlap: int  
):  
    try:  
        tasks[task_id] = {"status": "processing", "step": "Chunking content..."}  
        print_log(f"[Task {task_id}] Chunking '{source}'...")  
  
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  
        chunks = text_splitter.split_text(content)  
        if not chunks:  
            raise ValueError("Could not split content into any chunks.")  
  
        tasks[task_id] = {"status": "processing", "step": "Generating embeddings..."}  
        print_log(f"[Task {task_id}] Generating embeddings for {len(chunks)} chunks...")  
  
        all_embeddings = _embed_chunks_parallel(chunks)  
  
        tasks[task_id] = {"status": "processing", "step": "Saving to knowledge base..."}  
        print_log(f"[Task {task_id}] Saving {len(chunks)} chunks to the database...")  
  
        to_insert = []  
        for i, chunk_text in enumerate(chunks):  
            doc = {  
                "text": chunk_text,  
                "metadata": {  
                    "source": source,  
                    "source_type": source_type,  
                    SESSION_FIELD: session_id  
                }  
            }  
            for model_name, emb_list in all_embeddings.items():  
                if emb_list:  
                    vec_field = EMBEDDING_CONFIG[model_name]["vector_field"]  
                    doc[vec_field] = emb_list[i]  
            to_insert.append(doc)  
  
        config.collection.insert_many(to_insert)  
  
        final_message = f"Successfully ingested {len(chunks)} chunks from source '{source}'."  
        tasks[task_id] = {"status": "complete", "message": final_message}  
        print_log(f"[Task {task_id}] {final_message}")  
  
    except Exception as e:  
        error_message = f"Ingestion failed: {str(e)}"  
        print_log(f"[Task {task_id}] [ERROR] {error_message}\n{traceback.format_exc()}")  
        tasks[task_id] = {"status": "failed", "message": error_message}  
  
# --- 5. LANGCHAIN SETUP ---  
CHAT_DEPLOYMENT_NAME = os.getenv("CHAT_DEPLOYMENT_NAME", "gpt-4o")  
print_log(f"--- 🧠 Initializing LLM ---\nChat Deployment: '{CHAT_DEPLOYMENT_NAME}'\n------------------------------------")  
  
llm = AzureChatOpenAI(  
    azure_deployment=CHAT_DEPLOYMENT_NAME,  
    api_version="2024-02-01",  
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    temperature=0  
)  
  
# --- 6. CORE DATABASE & EMBEDDING LOGIC ---  
def _embed_chunks_parallel(chunks: List[str]) -> Dict[str, List[List[float]]]:  
    embeddings = {}  
  
    with concurrent.futures.ThreadPoolExecutor() as inner_executor:  
        future_to_model = {  
            inner_executor.submit(client.embed_documents, chunks): model_name  
            for model_name, client in config.embedding_clients.items()  
        }  
  
        for future in concurrent.futures.as_completed(future_to_model):  
            model_name = future_to_model[future]  
            try:  
                embeddings[model_name] = future.result()  
                print_log(f"[INFO] Successfully generated {len(chunks)} embeddings with {model_name}.")  
            except Exception as e:  
                print_log(f"[ERROR] Failed to generate embeddings with {model_name}: {e}")  
                embeddings[model_name] = None  
  
    return embeddings  
  
def _update_chunk_in_db(chunk_id: str, new_content: str) -> Dict[str, Any]:  
    oid = ObjectId(chunk_id)  
    update_payload = {"$set": {"text": new_content}}  
  
    print_log(f"[INFO] Re-embedding chunk {chunk_id} with all available models...")  
    all_embeddings = _embed_chunks_parallel([new_content])  
    for model_name, embeddings_list in all_embeddings.items():  
        if embeddings_list:  
            vector_field = EMBEDDING_CONFIG[model_name]["vector_field"]  
            update_payload["$set"][vector_field] = embeddings_list[0]  
  
    result = config.collection.update_one({"_id": oid}, update_payload)  
    if result.matched_count == 0:  
        raise ValueError(f"Could not find a chunk with ID '{chunk_id}'.")  
  
    return {"status": "success", "message": f"Chunk '{chunk_id}' updated (re-embedded)."}  
  
def _delete_chunk_from_db(chunk_id: str) -> Dict[str, Any]:  
    result = config.collection.delete_one({"_id": ObjectId(chunk_id)})  
    if result.deleted_count == 0:  
        raise ValueError(f"Could not find chunk ID '{chunk_id}' to delete.")  
    return {"status": "success", "message": f"Chunk '{chunk_id}' deleted."}  
  
def _perform_vector_search(query: str, session_id: str, embedding_model: str, num_sources: int) -> List[Dict]:  
    if embedding_model not in config.embedding_clients:  
        raise ValueError(f"Embedding model '{embedding_model}' is not available.")  
  
    model_config = EMBEDDING_CONFIG[embedding_model]  
    embedding_client = config.embedding_clients[embedding_model]  
    query_vector = embedding_client.embed_query(query)  
  
    pipeline = [  
        {  
            "$vectorSearch": {  
                "index": model_config['index_name'],  
                "path": model_config['vector_field'],  
                "queryVector": query_vector,  
                "numCandidates": num_sources * 10,  
                "limit": num_sources,  
                "filter": {  
                    f"metadata.{SESSION_FIELD}": {"$eq": session_id}  
                }  
            }  
        },  
        {  
            "$project": {  
                "_id": 0,  
                "content": "$text",  
                "source": "$metadata.source",  
                "score": {"$meta": "vectorSearchScore"}  
            }  
        }  
    ]  
  
    results = list(config.collection.aggregate(pipeline))  
    return results  
  
# --- 7. TOOL DEFINITIONS ---  
@tool  
def search_knowledge_base(query: str, embedding_model: str, num_sources: int = 3, max_chunk_length: int = 2000) -> str:  
    """Searches the knowledge base for relevant chunks given a query."""  
    try:  
        print_log(f"[INFO] Searching with model: {embedding_model}, k={num_sources}, max_length={max_chunk_length}")  
        results_with_scores = _perform_vector_search(  
            query, config.current_session, embedding_model, num_sources  
        )  
  
        if not results_with_scores:  
            config.last_retrieved_sources = []  
            return f"No relevant info found in session '{config.current_session}'."  
  
        sources_from_db = [r.get('source','N/A') for r in results_with_scores]  
        config.last_retrieved_sources = list(set(sources_from_db))  
  
        context_parts = []  
        for result in results_with_scores:  
            content = result.get('content','')  
            source = result.get('source','N/A')  
            score = result.get('score', 0.0)  
  
            if max_chunk_length > 0 and len(content) > max_chunk_length:  
                content = content[:max_chunk_length] + "... [truncated]"  
            context_parts.append(f"Source: {source} (Score: {score:.4f})\nContent: {content}")  
  
        context = "\n---\n".join(context_parts)  
        return f"Retrieved the following from '{embedding_model}':\n{context}"  
  
    except Exception as e:  
        config.last_retrieved_sources = []  
        print_log(f"[ERROR] Vector search failed: {e}")  
        return f"❌ Search error: {e}"  
  
@tool  
def read_url(url: str, chunk_size: int=1000, chunk_overlap: int=150) -> str:  
    """Add a publicly-accessible URL to the knowledge base."""  
    try:  
        if config.collection.find_one({"metadata.source": url, f"metadata.{SESSION_FIELD}": config.current_session}):  
            return f"❌ Source '{url}' already exists in session '{config.current_session}'."  
  
        jina_api_key = os.getenv("JINA_API_KEY")  
        if not jina_api_key:  
            return "❌ JINA_API_KEY is not set."  
  
        headers = {"Authorization": f"Bearer {jina_api_key}", "Accept": "application/json"}  
        print_log(f"[INFO] Reading & ingesting URL: {url}")  
  
        resp = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=30)  
        resp.raise_for_status()  
        page_content = resp.json().get('data', {}).get('content', "")  
        if not page_content:  
            return f"❌ No meaningful content from {url}."  
  
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  
        chunks = splitter.split_text(page_content)  
        if not chunks:  
            return "❌ Could not split into chunks."  
  
        all_embeddings = _embed_chunks_parallel(chunks)  
  
        docs_to_insert = []  
        for i, ctext in enumerate(chunks):  
            doc = {  
                "text": ctext,  
                "metadata": {  
                    "source": url,  
                    "source_type": "url",  
                    SESSION_FIELD: config.current_session  
                }  
            }  
            for model_name, embeddings_list in all_embeddings.items():  
                if embeddings_list:  
                    ffield = EMBEDDING_CONFIG[model_name]["vector_field"]  
                    doc[ffield] = embeddings_list[i]  
            docs_to_insert.append(doc)  
  
        config.collection.insert_many(docs_to_insert)  
        return f"✅ Ingested {len(chunks)} chunks from {url} into '{config.current_session}'."  
  
    except Exception as e:  
        print_log(f"[ERROR] Ingestion from URL failed: {e}\n{traceback.format_exc()}")  
        return f"❌ Unexpected error: {e}"  
  
@tool  
def update_chunk(chunk_id: str, new_content: str) -> str:  
    """Update the text (and embeddings) of a chunk by its ID."""  
    try:  
        res = _update_chunk_in_db(chunk_id, new_content)  
        return f"✅ {res['message']}"  
    except Exception as e:  
        return f"❌ Failed to update chunk. Error: {e}"  
  
@tool  
def delete_chunk(chunk_id: str) -> str:  
    """Delete a chunk from the knowledge base by ID."""  
    try:  
        res = _delete_chunk_from_db(chunk_id)  
        return f"✅ {res['message']}"  
    except Exception as e:  
        return f"❌ Failed to delete chunk. Error: {e}"  
  
@tool  
def switch_session(session_id: str) -> str:  
    """Switch to another session."""  
    config.current_session = session_id  
    if session_id not in config.chat_history:  
        config.chat_history[session_id] = []  
    return f"✅ Switched to session: **{session_id}**."  
  
@tool  
def create_session(session_id: str) -> str:  
    """Create and switch to a new session."""  
    existing_sessions = config.collection.distinct(f"metadata.{SESSION_FIELD}")  
    if session_id in existing_sessions:  
        return f"❌ Session **'{session_id}'** already exists."  
  
    config.current_session = session_id  
    config.chat_history[session_id] = []  
    return f"✅ Created and switched to new session: **{session_id}**."  
  
@tool  
def list_sources() -> str:  
    """List all sources in the current session."""  
    sources = config.collection.distinct("metadata.source", {f"metadata.{SESSION_FIELD}": config.current_session})  
    if not sources:  
        return f"No sources found in session '{config.current_session}'."  
    return "Sources:\n" + "\n".join([f"- {s}" for s in sources])  
  
@tool  
def remove_all_sources() -> str:  
    """Remove everything in the current session."""  
    r = config.collection.delete_many({f"metadata.{SESSION_FIELD}": config.current_session})  
    return f"🗑 Removed all docs from session '{config.current_session}' (deleted {r.deleted_count})."  
  
# --- 8. AGENT AND PROMPT SETUP ---  
tools = [  
    search_knowledge_base,  
    switch_session,  
    create_session,  
    list_sources,  
    remove_all_sources,  
    update_chunk,  
    delete_chunk,  
    read_url  
]  
  
AGENT_SYSTEM_PROMPT = (  
    "You are a helpful AI assistant with a knowledge base. Your primary tool is 'search_knowledge_base'. "  
    "When the user says 'num_sources' or 'max_chunk_length', pass them along to the tool. "  
    "Always use 'embedding_model' from the user's query. For other tasks, use the appropriate tool."  
)  
  
prompt = ChatPromptTemplate.from_messages([  
    ("system", AGENT_SYSTEM_PROMPT),  
    MessagesPlaceholder(variable_name="chat_history"),  
    ("human", "{input}"),  
    MessagesPlaceholder(variable_name="agent_scratchpad")  
])  
  
agent = create_openai_tools_agent(llm, tools, prompt)  
agent_executor = AgentExecutor(  
    agent=agent,  
    tools=tools,  
    verbose=True,  
    return_intermediate_steps=True  
)  
  
# --- 9. FLASK APP ---  
app = Flask(__name__, template_folder="templates", static_folder="static")

CORS(app)  
  
@app.route("/")
def index():
    # Use render_template to serve the index.html file
    return render_template("index.html")
  
@app.route("/ingest", methods=["POST"])  
def start_ingestion_task():  
    data = request.json  
    content = data.get("content")  
    source = data.get("source")  
    source_type = data.get("source_type", "unknown")  
    session_id = data.get("session_id")  
    chunk_size = data.get("chunk_size", 1000)  
    chunk_overlap = data.get("chunk_overlap", 150)  
  
    if not all([content, source, session_id]):  
        return jsonify({"error": "Missing required fields."}), 400  
  
    try:  
        if config.collection.count_documents(  
            {"metadata.source": source, f"metadata.{SESSION_FIELD}": session_id},  
            limit=1  
        ) > 0:  
            return jsonify({"error": f"Source '{source}' already exists in session '{session_id}'."}), 409  
    except Exception as e:  
        return jsonify({"error": f"Database error: {str(e)}"}), 500  
  
    task_id = str(uuid.uuid4())  
    tasks[task_id] = {"status": "pending"}  
  
    executor.submit(  
        run_ingestion_task,  
        task_id,  
        content,  
        source,  
        source_type,  
        session_id,  
        chunk_size,  
        chunk_overlap  
    )  
  
    return jsonify({"task_id": task_id}), 202  
  
@app.route("/ingest/status/<task_id>", methods=["GET"])  
def get_ingestion_status(task_id):  
    if task_id not in tasks:  
        return jsonify({"status": "not_found"}), 200  
    return jsonify(tasks[task_id]), 200  
  
@app.route("/chat", methods=["POST"])  
def chat():  
    data = request.json  
    user_input = data.get("query")  
    session_id = data.get("session_id")  
    embedding_model = data.get("embedding_model", "openai")  
    rag_params = data.get("rag_params", {})  
    num_sources = rag_params.get("num_sources", config.rag_config["num_sources"])  
    max_chunk_length = rag_params.get("max_chunk_length", config.rag_config["max_chunk_length"])  
  
    if not user_input or not session_id:  
        return jsonify({"error": "Missing 'query' or 'session_id'"}), 400  
  
    augmented_input = (  
        f"Using retrieval settings (num_sources={num_sources}, max_chunk_length={max_chunk_length}, "  
        f"embedding_model='{embedding_model}'), answer: {user_input}"  
    )  
    print_log(f"\n--- Turn (Session: {session_id}) ---\nAugmented >>> {augmented_input}")  
  
    original_session = config.current_session  
    if session_id not in config.chat_history:  
        config.chat_history[session_id] = []  
    try:  
        config.current_session = session_id  
        current_chat_history = config.chat_history[session_id]  
        if len(current_chat_history) > 10:  
            current_chat_history = current_chat_history[-10:]  
  
        config.last_retrieved_sources = []  
  
        response = agent_executor.invoke({  
            "input": augmented_input,  
            "chat_history": current_chat_history  
        })  
  
        current_chat_history.extend([  
            HumanMessage(content=user_input),  
            AIMessage(content=response["output"])  
        ])  
        config.chat_history[session_id] = current_chat_history  
  
        messages = [{"type": "bot-message", "content": response["output"]}]  
        sources_used = config.last_retrieved_sources  
  
        resp_data = {  
            "messages": messages,  
            "sources": sources_used  
        }  
  
        all_sessions = set(config.collection.distinct(f"metadata.{SESSION_FIELD}") or ["default"])  
        all_sessions.add(session_id)  
        resp_data["session_update"] = {  
            "all_sessions": sorted(list(all_sessions)),  
            "current_session": session_id  
        }  
  
        return jsonify(resp_data)  
    except Exception as e:  
        print_log(f"[ERROR] agent invoke: {e}\n{traceback.format_exc()}")  
        return jsonify({"error": str(e)}), 500  
    finally:  
        config.current_session = original_session  
  
@app.route("/state", methods=["GET"])  
def get_state():  
    sessions = set(config.collection.distinct(f"metadata.{SESSION_FIELD}") or ["default"])  
    available_models = list(config.embedding_clients.keys())  
    return jsonify({  
        "all_sessions": sorted(list(sessions)),  
        "current_session": config.current_session,  
        "available_embedding_models": available_models  
    })  
  
@app.route("/history/clear", methods=["POST"])  
def clear_history():  
    data = request.json  
    session_id = data.get("session_id")  
    if not session_id:  
        return jsonify({"error": "Missing 'session_id'"}), 400  
  
    if session_id in config.chat_history:  
        config.chat_history[session_id] = []  
        print_log(f"[INFO] Cleared chat history for session: {session_id}")  
        return jsonify({"status": "success", "message": f"Chat history for '{session_id}' cleared."})  
  
    return jsonify({"status": "not_found", "message": f"Session '{session_id}' not found."}), 404  
  
@app.route("/preview_search", methods=["POST"])  
def preview_search():  
    data = request.json  
    query = data.get("query")  
    session_id = data.get("session_id")  
    embedding_model = data.get("embedding_model", "openai")  
    num_sources = data.get("num_sources", 3)  
    if not query or not session_id:  
        return jsonify({"error": "Missing required fields"}), 400  
    try:  
        results = _perform_vector_search(query, session_id, embedding_model, num_sources)  
        return jsonify(results)  
    except Exception as e:  
        print_log(f"[ERROR] Preview search failed: {e}")  
        return jsonify({"error": str(e)}), 500  
  
@app.route("/preview_file", methods=["POST"])  
def preview_file():  
    if "file" not in request.files:  
        return jsonify({"error": "No file part"}), 400  
    file = request.files["file"]  
    if file.filename == "":  
        return jsonify({"error": "No selected file"}), 400  
  
    _, extension = os.path.splitext(file.filename.lower())  
    MAX_PREVIEW = 50000  
  
    if extension in [".txt", ".md"]:  
        text_data = file.read().decode("utf-8", errors="replace")  
        if len(text_data) > MAX_PREVIEW:  
            text_data = text_data[:MAX_PREVIEW] + "\n\n[TRUNCATED]"  
        return jsonify({"content": text_data, "filename": file.filename})  
  
    temp_file_path = ""  
    try:  
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:  
            file.save(tmp.name)  
            temp_file_path = tmp.name  
  
        converter = DocumentConverter()  
        result = converter.convert(temp_file_path)  
        doc_text = result.document.export_to_markdown()  
  
        if len(doc_text) > MAX_PREVIEW:  
            doc_text = doc_text[:MAX_PREVIEW] + "\n\n[TRUNCATED]"  
  
        return jsonify({  
            "content": doc_text,  
            "filename": file.filename  
        })  
    finally:  
        if temp_file_path and os.path.exists(temp_file_path):  
            os.unlink(temp_file_path)  
  
@app.route("/preview_url", methods=["GET"])  
def preview_url():  
    url = request.args.get("url")  
    if not url:  
        return jsonify({"error": "URL parameter is required."}), 400  
  
    jina_api_key = os.getenv("JINA_API_KEY")  
    if not jina_api_key:  
        return jsonify({"error": "JINA_API_KEY not set."}), 500  
  
    headers = {"Authorization": f"Bearer {jina_api_key}", "Accept": "application/json"}  
    try:  
        print_log(f"[INFO] Previewing URL: {url}")  
        resp = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=30)  
        resp.raise_for_status()  
        page_content = resp.json().get("data", {}).get("content", "")  
        MAX_PREVIEW = 50000  
        if len(page_content) > MAX_PREVIEW:  
            page_content = page_content[:MAX_PREVIEW] + "\n\n[TRUNCATED]"  
        return jsonify({"markdown": page_content})  
    except requests.exceptions.RequestException as e:  
        return jsonify({"error": f"Error fetching URL content: {e}"}), 500  
    except Exception as e:  
        return jsonify({"error": f"Unexpected error: {e}"}), 500  
  
@app.route("/search", methods=["POST"])  
def search_web():  
    data = request.json  
    query = data.get("query")  
    num_results = data.get("num_results", 5)  
    if not query:  
        return jsonify({"error": "Query is required"}), 400  
    try:  
        print_log(f"[INFO] Web search for: '{query}'")  
        with DDGS() as ddgs:  
            results = [r for r in ddgs.text(query, max_results=num_results)]  
        return jsonify(results)  
    except Exception as e:  
        print_log(f"[ERROR] Web search failed: {e}\n{traceback.format_exc()}")  
        return jsonify({"error": f"Web search error: {str(e)}"}), 500  
  
@app.route("/sources", methods=["GET"])  
def get_sources():  
    session_id = request.args.get("session_id", "default")  
    pipeline = [  
        {"$match": {f"metadata.{SESSION_FIELD}": session_id}},  
        {"$group": {"_id": "$metadata.source", "source_type": {"$first": "$metadata.source_type"}}},  
        {"$project": {"name": "$_id", "type": {"$ifNull": ["$source_type", "unknown"]}, "_id": 0}},  
        {"$sort": {"name": 1}}  
    ]  
    return jsonify(list(config.collection.aggregate(pipeline)))  
  
@app.route("/chunks", methods=["GET"])  
def get_chunks():  
    session_id = request.args.get("session_id", "default")  
    source_url = request.args.get("source_url")  
    if not source_url:  
        return jsonify({"error": "source_url required"}), 400  
  
    cursor = config.collection.find(  
        {"metadata.source": source_url, f"metadata.{SESSION_FIELD}": session_id},  
        {"text": 1, "_id": 1}  
    )  
    return jsonify([{"_id": str(c["_id"]), "text": c["text"]} for c in cursor])  
  
@app.route("/chunk/<chunk_id>", methods=["DELETE"])  
def api_delete_chunk(chunk_id):  
    try:  
        return jsonify(_delete_chunk_from_db(chunk_id))  
    except Exception as e:  
        return jsonify({"error": str(e)}), 500  
  
@app.route("/chunk/<chunk_id>", methods=["PUT"])  
def api_update_chunk(chunk_id):  
    new_content = request.json.get("content")  
    if not new_content:  
        return jsonify({"error": "New content is required"}), 400  
  
    try:  
        return jsonify(_update_chunk_in_db(chunk_id, new_content))  
    except Exception as e:  
        return jsonify({"error": str(e)}), 500  
  
# --- 10. STARTUP & DB SETUP ---  
def setup_database_and_index():  
    print_log("--- 🚀 Initializing DB and Vector Search Indexes ---")  
    if COLLECTION_NAME not in config.db.list_collection_names():  
        config.db.create_collection(COLLECTION_NAME)  
  
    for model_name, model_cfg in EMBEDDING_CONFIG.items():  
        if model_name not in config.embedding_clients:  
            print_log(f"[INFO] Model '{model_name}' not loaded; skipping index creation.")  
            continue  
  
        index_name = model_cfg["index_name"]  
        vector_field = model_cfg["vector_field"]  
        dims = model_cfg["dimensions"]  
  
        definition = {  
            "fields": [  
                {  
                    "type": "vector",  
                    "path": vector_field,  
                    "numDimensions": dims,  
                    "similarity": "cosine"  
                },  
                {  
                    "type": "filter",  
                    "path": f"metadata.{SESSION_FIELD}"  
                }  
            ]  
        }  
  
        try:  
            existing = next(config.collection.list_search_indexes(name=index_name), None)  
            if not existing:  
                print_log(f"[ACTION] Creating index '{index_name}' for '{model_name}'...")  
                config.collection.create_search_index(  
                    model=SearchIndexModel(name=index_name, type="vectorSearch", definition=definition)  
                )  
                print_log(f"[INFO] Index creation for '{index_name}' started.")  
            else:  
                print_log(f"[INFO] Index '{index_name}' already exists. ✅")  
        except OperationFailure as e:  
            if "already exists" in str(e).lower():  
                print_log(f"[INFO] Index '{index_name}' already exists. ✅")  
            else:  
                print_log(f"[ERROR] Creating index '{index_name}' failed: {e}")  
                raise  
  
if __name__ == "__main__":  
    setup_database_and_index()  
    print_log("--- ✅ Setup complete. Starting server at http://127.0.0.1:5001 ---")  
    app.run(debug=True, port=5001)  