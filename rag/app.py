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

# --- 0. DEPENDENCIES ---
# pip install pymongo flask openai python-dotenv flask-cors requests langchain
# pip install langchain-openai langchain-mongodb ddgs docling langchain-voyageai voyageai
# REMEMBER: if previously installed "duckduckgo-search", remove it:
#    pip uninstall duckduckgo-search -y
# then "pip install ddgs".

# --- Core Libraries ---
import pymongo
from flask import Flask, request, jsonify, render_template, render_template_string
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

# --- 1. CONFIG / ENV SETUP ---
load_dotenv()

DATABASE_NAME = "interactive_rag_db"
COLLECTION_NAME = "knowledge_base_sessions"
SESSION_FIELD = "session_id"

# Embedding configuration for multiple models
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

# Logging setup
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


# --- 2. GLOBAL AGENT STATE ---
class AgentConfig:
    def __init__(self):
        self.rag_config = {
            "num_sources": 3,
            "min_rel_score": 0.0,
            "max_chunk_length": 2000
        }
        self.embedding_clients = {}

        # Connect to MongoDB
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

        # In-memory chat logs
        self.chat_history = {}
        self.current_session = "default"

        # For storing last search result sources (optional)
        self.last_retrieved_sources = []

        # Initialize embeddings
        print_log("--- 🧠 Initializing Embedding Clients ---")
        # 1) OpenAI
        self.embedding_clients["openai"] = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        print_log("[INFO] OpenAI embedding client initialized.")

        # 2) VoyageAI (if VOYAGE_API_KEY is set)
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


# --- 3. BACKGROUND TASK SETUP ---
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
    """Handles chunking & embedding in a background thread."""
    try:
        tasks[task_id] = {"status": "processing", "step": "Chunking content..."}
        print_log(f"[Task {task_id}] Chunking '{source}' with size {chunk_size} and overlap {chunk_overlap}...")

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
                    SESSION_FIELD: session_id,
                    "chunk_index": i
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


# --- 4. LANGCHAIN SETUP ---
CHAT_DEPLOYMENT_NAME = os.getenv("CHAT_DEPLOYMENT_NAME", "gpt-4o")
print_log(f"--- 🧠 Initializing LLM ---\nChat Deployment: '{CHAT_DEPLOYMENT_NAME}'\n------------------------------------")

llm = AzureChatOpenAI(
    azure_deployment=CHAT_DEPLOYMENT_NAME,
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)


# --- 5. CORE FUNCTIONS ---
def _embed_chunks_parallel(chunks: List[str]) -> Dict[str, List[List[float]]]:
    """Embed the text chunks in parallel for each available embedding model."""
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
                print_log(f"[INFO] Generated {len(chunks)} embeddings with {model_name}.")
            except Exception as e:
                print_log(f"[ERROR] Embedding with {model_name} failed: {e}")
                embeddings[model_name] = None
    return embeddings

def _update_chunk_in_db(chunk_id: str, new_content: str) -> Dict[str, Any]:
    """Update chunk text and re-embed with all available models."""
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
        raise ValueError(f"Could not find chunk with ID '{chunk_id}'.")

    return {"status": "success", "message": f"Chunk '{chunk_id}' updated (re-embedded)."}

def _delete_chunk_from_db(chunk_id: str) -> Dict[str, Any]:
    """Delete a single chunk by ID."""
    result = config.collection.delete_one({"_id": ObjectId(chunk_id)})
    if result.deleted_count == 0:
        raise ValueError(f"Could not find chunk '{chunk_id}' to delete.")
    return {"status": "success", "message": f"Chunk '{chunk_id}' deleted."}

def _perform_vector_search(
    query: str,
    session_id: str,
    embedding_model: str,
    num_sources: int
) -> List[Dict]:
    """Perform a vector-based search in MongoDB for top `num_sources` results."""
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

    return list(config.collection.aggregate(pipeline))


# --- 6. AGENT TOOLS ---
@tool
def search_knowledge_base(query: str, embedding_model: str, num_sources: int = 3, max_chunk_length: int = 2000) -> str:
    """Query the knowledge base to find relevant chunks for `query`."""
    try:
        print_log(f"[INFO] Searching with '{embedding_model}' → top {num_sources}")
        results_with_scores = _perform_vector_search(query, config.current_session, embedding_model, num_sources)

        if not results_with_scores:
            config.last_retrieved_sources = []
            return f"No relevant info found in session '{config.current_session}'."

        # Remember sources
        found_sources = [r.get("source", "N/A") for r in results_with_scores]
        config.last_retrieved_sources = list(set(found_sources))

        # Build a context string
        context_parts = []
        for r in results_with_scores:
            text = r.get("content", "")
            src = r.get("source", "N/A")
            score = r.get("score", 0.0)
            if max_chunk_length and len(text) > max_chunk_length:
                text = text[:max_chunk_length] + "... [truncated]"
            context_parts.append(f"Source: {src} (Score: {score:.4f})\nContent: {text}")

        context = "\n---\n".join(context_parts)
        return f"Retrieved from '{embedding_model}':\n{context}"

    except Exception as e:
        config.last_retrieved_sources = []
        print_log(f"[ERROR] search_knowledge_base: {e}")
        return f"❌ Search error: {e}"

@tool
def read_url(url: str, chunk_size: int=1000, chunk_overlap: int=150) -> str:
    """Adds a URL's content (via r.jina.ai) into the knowledge base."""
    try:
        if config.collection.find_one({"metadata.source": url, f"metadata.{SESSION_FIELD}": config.current_session}):
            return f"❌ Source '{url}' already exists in session '{config.current_session}'."

        jina_key = os.getenv("JINA_API_KEY")
        if not jina_key:
            return "❌ JINA_API_KEY not set."

        headers = {"Authorization": f"Bearer {jina_key}", "Accept": "application/json"}
        print_log(f"[INFO] Reading & ingesting URL: {url}")

        resp = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=30)
        resp.raise_for_status()
        page_content = resp.json().get("data", {}).get("content", "")
        if not page_content:
            return f"❌ No meaningful content from {url}."

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(page_content)
        if not chunks:
            return "❌ Could not split content into chunks."

        all_embeddings = _embed_chunks_parallel(chunks)
        docs_to_insert = []
        for i, ctext in enumerate(chunks):
            doc = {
                "text": ctext,
                "metadata": {
                    "source": url,
                    "source_type": "url",
                    SESSION_FIELD: config.current_session,
                    "chunk_index": i
                }
            }
            for model_name, embed_list in all_embeddings.items():
                if embed_list:
                    vector_field = EMBEDDING_CONFIG[model_name]["vector_field"]
                    doc[vector_field] = embed_list[i]
            docs_to_insert.append(doc)

        config.collection.insert_many(docs_to_insert)
        return f"✅ Ingested {len(chunks)} chunks from {url} into '{config.current_session}'."

    except Exception as e:
        print_log(f"[ERROR] read_url: {e}\n{traceback.format_exc()}")
        return f"❌ Ingestion error: {e}"

@tool
def update_chunk(chunk_id: str, new_content: str) -> str:
    """Updates chunk text (and embeddings) by chunk ID."""
    try:
        res = _update_chunk_in_db(chunk_id, new_content)
        return f"✅ {res['message']}"
    except Exception as e:
        return f"❌ Failed to update chunk: {e}"

@tool
def delete_chunk(chunk_id: str) -> str:
    """Deletes a chunk from the knowledge base by ID."""
    try:
        res = _delete_chunk_in_db(chunk_id)
        return f"✅ {res['message']}"
    except Exception as e:
        return f"❌ Failed to delete chunk: {e}"

@tool
def switch_session(session_id: str) -> str:
    """Switch to another session in memory."""
    config.current_session = session_id
    if session_id not in config.chat_history:
        config.chat_history[session_id] = []
    return f"✅ Switched to session: **{session_id}**."

@tool
def create_session(session_id: str) -> str:
    """Create a new session in memory only (no marker doc)."""
    existing_sessions = config.collection.distinct(f"metadata.{SESSION_FIELD}")
    if session_id in existing_sessions:
        return f"❌ Session **'{session_id}'** already exists."

    config.current_session = session_id
    if session_id not in config.chat_history:
        config.chat_history[session_id] = []
    return f"✅ Created and switched to new session: **{session_id}**."

@tool
def list_sources() -> str:
    """List all sources in the current session."""
    sources = config.collection.distinct("metadata.source", {f"metadata.{SESSION_FIELD}": config.current_session})
    if not sources:
        return f"No sources found in session '{config.current_session}'."
    return "Sources:\n" + "\n".join(f"- {s}" for s in sources)

@tool
def remove_all_sources() -> str:
    """Remove all docs from the current session."""
    r = config.collection.delete_many({f"metadata.{SESSION_FIELD}": config.current_session})
    return f"🗑 Removed all docs from session '{config.current_session}' (deleted {r.deleted_count})."

# --- 7. AGENT PROMPT + EXECUTOR ---
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

available_model_keys = list(config.embedding_clients.keys())
AGENT_SYSTEM_PROMPT = (
    "You are an AI assistant designed to answer questions using a private knowledge base. "
    "Your primary directive is to **ALWAYS use the `search_knowledge_base` tool** to find relevant information before answering any user query. "
    "**Do not answer from your general knowledge.** Your answers must be based *only* on the context provided by the `search_knowledge_base` tool. "
    "If the tool returns no relevant information or the context is insufficient, you MUST state that you could not find an answer in the knowledge base. "
    f"The available `embedding_model` options for the search tool are: {', '.join(available_model_keys)}. "
    "For other tasks like managing sessions or sources, use the appropriate tool."
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


# --- 8. FLASK APP ---
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

# ---- Ingestion Endpoints ----
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

    # Check duplicates
    if config.collection.count_documents(
        {"metadata.source": source, f"metadata.{SESSION_FIELD}": session_id},
        limit=1
    ) > 0:
        return jsonify({"error": f"Source '{source}' already exists in session '{session_id}'."}), 409

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

# ---- Chat Endpoint ----
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

    print_log(f"\n--- Turn for session '{session_id}' ---\n")
    original_session = config.current_session

    try:
        # Switch session in memory
        config.current_session = session_id

        # Initialize chat in memory if needed
        if session_id not in config.chat_history:
            config.chat_history[session_id] = []

        # Shorten chat history if too long
        current_chat_history = config.chat_history[session_id]
        if len(current_chat_history) > 10:
            current_chat_history = current_chat_history[-10:]

        agent_input_string = (
            f"User query: '{user_input}'.\n\n"
            f"IMPORTANT INSTRUCTION: When you call the 'search_knowledge_base' tool, "
            f"you MUST set the 'embedding_model' parameter to '{embedding_model}'."
        )

        # Agent call
        response = agent_executor.invoke({
            "input": agent_input_string,
            "chat_history": current_chat_history,
            "num_sources": num_sources,
            "max_chunk_length": max_chunk_length
        })

        # Record the conversation
        current_chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response["output"])
        ])
        config.chat_history[session_id] = current_chat_history
        
        sources_used = config.last_retrieved_sources

        messages = [{
            "type": "bot-message",
            "content": response["output"],
            "sources": sources_used
        }]

        db_sessions = set(config.collection.distinct(f"metadata.{SESSION_FIELD}") or ["default"])
        mem_sessions = set(config.chat_history.keys())
        all_sessions = db_sessions.union(mem_sessions)

        resp_data = {
            "messages": messages,
            "session_update": {
                "all_sessions": sorted(list(all_sessions)),
                "current_session": config.current_session
            }
        }
        return jsonify(resp_data)

    except Exception as e:
        print_log(f"[ERROR] chat endpoint: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

    finally:
        pass

# ---- Session / State Endpoints ----
@app.route("/state", methods=["GET"])
def get_state():
    db_sessions = set(config.collection.distinct(f"metadata.{SESSION_FIELD}") or ["default"])
    mem_sessions = set(config.chat_history.keys()) or {"default"}
    all_sessions = db_sessions.union(mem_sessions)

    return jsonify({
        "all_sessions": sorted(list(all_sessions)),
        "current_session": config.current_session,
        "available_embedding_models": list(config.embedding_clients.keys())
    })

@app.route("/history/clear", methods=["POST"])
def clear_history():
    data = request.json
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing 'session_id'"}), 400

    if session_id in config.chat_history:
        config.chat_history[session_id] = []
        msg = f"Chat history for '{session_id}' cleared."
        print_log("[INFO] " + msg)
        return jsonify({"status": "success", "message": msg})

    return jsonify({"status": "not_found", "message": f"Session '{session_id}' not found."}), 404

# ---- Searching / Preview Endpoints ----
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
        print_log(f"[ERROR] preview_search: {e}")
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

    jina_key = os.getenv("JINA_API_KEY")
    if not jina_key:
        return jsonify({"error": "JINA_API_KEY not set."}), 500

    headers = {"Authorization": f"Bearer {jina_key}", "Accept": "application/json"}
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

# ---- Chunk Editing ----
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

# ---- Source Browsing ----
@app.route("/sources", methods=["GET"])
def get_sources():
    session_id = request.args.get("session_id", "default")
    pipeline = [
        {"$match": {f"metadata.{SESSION_FIELD}": session_id}},
        {
            "$group": {
                "_id": "$metadata.source",
                "source_type": {"$first": "$metadata.source_type"},
                "chunk_count": {"$sum": 1}
            }
        },
        {
            "$project": {
                "name": "$_id",
                "type": {"$ifNull": ["$source_type", "unknown"]},
                "chunk_count": "$chunk_count",
                "_id": 0
            }
        },
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
        {"_id": 1, "text": 1}
    )
    return jsonify([
        {"_id": str(doc["_id"]), "text": doc["text"]}
        for doc in cursor
    ])

# --- MODIFIED: Endpoint now returns a readable HTML page ---
@app.route("/source_content", methods=["GET"])
def get_source_content():
    session_id = request.args.get("session_id")
    source = request.args.get("source")

    if not session_id or not source:
        return "<h1>Error</h1><p>Missing 'session_id' or 'source' parameter.</p>", 400
    
    try:
        chunks_cursor = config.collection.find(
            {
                f"metadata.{SESSION_FIELD}": session_id,
                "metadata.source": source
            },
            {"text": 1, "_id": 0}
        ).sort("metadata.chunk_index", pymongo.ASCENDING)

        full_content = "".join([chunk.get('text', '') for chunk in chunks_cursor])

        if not full_content:
            return "<h1>Error</h1><p>Source not found or has no content.</p>", 404

        # Return a simple, styled HTML page instead of JSON
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>{{ source_name }}</title>
            <style>
                body { 
                    background-color: #121826; 
                    color: #e5e7eb; 
                    font-family: sans-serif; 
                    line-height: 1.6;
                    margin: 0;
                    padding: 2rem; 
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                }
                pre { 
                    white-space: pre-wrap; 
                    word-wrap: break-word; 
                    font-family: monospace; 
                    font-size: 1rem; 
                    background-color: #1d2333;
                    padding: 1.5rem;
                    border-radius: 8px;
                    border: 1px solid #333c51;
                }
                h1 { color: #00ED64; word-break: break-all; }
                a { color: #00ED64; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Source</h1>
                <p style="word-break: break-all;">
                    <a href="{{ source_name }}" target="_blank">{{ source_name }}</a>
                </p>
                <hr style="border-color: #333c51; margin: 1.5rem 0;">
                <pre>{{ content }}</pre>
            </div>
        </body>
        </html>
        """
        return render_template_string(html_template, source_name=source, content=full_content)

    except Exception as e:
        print_log(f"[ERROR] /source_content: {e}\n{traceback.format_exc()}")
        return f"<h1>Error</h1><p>An unexpected error occurred: {str(e)}</p>", 500


# --- 9. DB INDEX SETUP & LAUNCH ---
def setup_database_and_index():
    print_log("--- 🚀 Initializing DB and Vector Search Indexes ---")
    if COLLECTION_NAME not in config.db.list_collection_names():
        config.db.create_collection(COLLECTION_NAME)

    for model_name, model_cfg in EMBEDDING_CONFIG.items():
        if model_name not in config.embedding_clients:
            print_log(f"[WARN] Model '{model_name}' is not loaded, skipping index creation.")
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
                print_log(f"[ACTION] Creating index '{index_name}' for model '{model_name}'...")
                config.collection.create_search_index(
                    model=SearchIndexModel(name=index_name, type="vectorSearch", definition=definition)
                )
                print_log(f"[INFO] Finished creating index '{index_name}'.")
            else:
                print_log(f"[INFO] Index '{index_name}' already exists.")
        except OperationFailure as e:
            if "already exists" in str(e).lower():
                print_log(f"[INFO] Index '{index_name}' already exists. OK.")
            else:
                print_log(f"[ERROR] Creating index '{index_name}' failed: {e}")
                raise

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
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        print_log(f"[ERROR] Web search failed: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Web search error: {str(e)}"}), 500

@app.route("/chunk_preview", methods=["POST"])
def chunk_preview():
    data = request.json
    content = data.get("content")
    chunk_size = data.get("chunk_size", 1000)
    chunk_overlap = data.get("chunk_overlap", 150)

    if not content:
        return jsonify({"error": "Content is required"}), 400
    
    if chunk_overlap >= chunk_size:
        return jsonify({"error": "Chunk overlap must be smaller than chunk size."}), 400

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(content)
        return jsonify({"chunks": chunks})
    except Exception as e:
        print_log(f"[ERROR] Chunk preview failed: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    setup_database_and_index()
    print_log("--- ✅ Setup complete. Starting server at http://127.0.0.1:5001 ---")
    app.run(debug=True, port=5001)