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
# pip install pymongo flask openai python-dotenv flask-cors requests langchain firecrawl-py
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
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIG / ENV SETUP ---
# Try to load .env from multiple possible locations
env_loaded = load_dotenv()
if not env_loaded:
    # Try loading from parent directory (in case running from rag/ subdirectory)
    import pathlib
    parent_env = pathlib.Path(__file__).parent.parent / ".env"
    if parent_env.exists():
        load_dotenv(parent_env)
        print(f"[INFO] Loaded .env from parent directory: {parent_env}")
    else:
        print(f"[WARN] .env file not found. Make sure it exists in the project root or rag/ directory.")
else:
    print(f"[INFO] .env file loaded successfully.")

# Debug: Verify FIRECRAWL_API_KEY is accessible
firecrawl_key_check = os.getenv("FIRECRAWL_API_KEY")
if firecrawl_key_check:
    print(f"[DEBUG] FIRECRAWL_API_KEY found (length: {len(firecrawl_key_check)}, starts with: {firecrawl_key_check[:5]}...)")
else:
    print(f"[DEBUG] FIRECRAWL_API_KEY NOT found in environment variables.")

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
                os.getenv("MONGO_URI"),
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
        
        # Debug: Store recent LLM requests and retrieved chunks
        self.debug_requests = []  # List of {timestamp, query, model, sources, response_length}
        self.debug_retrieved_chunks = []  # List of {timestamp, query, chunks: [{text, source, score}]}

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
        
        # Firecrawl API key (we'll use HTTP API directly, no SDK needed)
        self.firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        if self.firecrawl_api_key:
            self.firecrawl_api_key = self.firecrawl_api_key.strip()
            print_log("[INFO] FIRECRAWL_API_KEY found. URL scraping via Firecrawl HTTP API enabled.")
        else:
            firecrawl_key_raw = os.getenv("FIRECRAWL_API_KEY", "NOT_SET")
            print_log(f"[WARN] FIRECRAWL_API_KEY not found or empty. Raw value: '{firecrawl_key_raw[:20]}...' (first 20 chars). URL scraping will fail.")

        print_log("------------------------------------")

config = AgentConfig()


# --- 3. BACKGROUND TASK SETUP ---
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
tasks = {}
# Track which indexes are being created to prevent duplicate attempts
index_creation_in_progress = set()

def _ensure_indexes_ready_for_ingestion(max_wait: int = 60):
    """
    Ensures at least one vector search index is ready before ingestion.
    Returns True if at least one index is ready, False otherwise.
    """
    ready_indexes = []
    for model_name, model_cfg in EMBEDDING_CONFIG.items():
        if model_name not in config.embedding_clients:
            continue
        
        index_name = model_cfg["index_name"]
        existing = _get_search_index(config.collection, index_name)
        if existing and existing.get("queryable"):
            ready_indexes.append(index_name)
    
    if ready_indexes:
        print_log(f"[INFO] Found {len(ready_indexes)} ready index(es): {', '.join(ready_indexes)}")
        return True
    
    # No indexes ready, wait a bit for them to become ready
    print_log(f"[WARN] No vector search indexes are queryable yet. Waiting up to {max_wait}s...")
    start_time = time.time()
    poll_interval = 2
    
    while time.time() - start_time < max_wait:
        for model_name, model_cfg in EMBEDDING_CONFIG.items():
            if model_name not in config.embedding_clients:
                continue
            
            index_name = model_cfg["index_name"]
            existing = _get_search_index(config.collection, index_name)
            if existing and existing.get("queryable"):
                print_log(f"[INFO] Index '{index_name}' is now ready.")
                return True
        
        time.sleep(poll_interval)
    
    print_log(f"[WARN] No indexes became ready within {max_wait}s. Proceeding with ingestion anyway (indexes may still be building).")
    return False

def run_ingestion_task(
    task_id: str,
    content: str,
    source: str,
    source_type: str,
    session_id: str,
    chunk_size: int,
    chunk_overlap: int
):
    """Handles chunking & embedding in a background thread with robust error handling."""
    try:
        tasks[task_id] = {"status": "processing", "step": "Chunking content..."}
        print_log(f"[Task {task_id}] Starting ingestion for '{source}' (session: {session_id})...")

        # Step 1: Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(content)
        if not chunks:
            raise ValueError("Could not split content into any chunks.")
        print_log(f"[Task {task_id}] Split content into {len(chunks)} chunks.")

        # Step 2: Generate embeddings
        tasks[task_id] = {"status": "processing", "step": "Generating embeddings..."}
        print_log(f"[Task {task_id}] Generating embeddings for {len(chunks)} chunks...")
        all_embeddings = _embed_chunks_parallel(chunks)
        
        # Verify we have at least one set of embeddings
        if not any(emb_list for emb_list in all_embeddings.values()):
            raise ValueError("Failed to generate embeddings for any model.")

        # Step 3: Ensure indexes are ready (non-blocking check)
        tasks[task_id] = {"status": "processing", "step": "Verifying vector search indexes..."}
        _ensure_indexes_ready_for_ingestion(max_wait=30)  # Quick check, don't block too long

        # Step 4: Prepare documents for insertion
        tasks[task_id] = {"status": "processing", "step": "Preparing documents..."}
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
            # Add embeddings for each available model
            for model_name, emb_list in all_embeddings.items():
                if emb_list and i < len(emb_list):
                    vec_field = EMBEDDING_CONFIG[model_name]["vector_field"]
                    doc[vec_field] = emb_list[i]
            to_insert.append(doc)

        # Step 5: Insert into database with retry logic
        tasks[task_id] = {"status": "processing", "step": "Saving to knowledge base..."}
        print_log(f"[Task {task_id}] Inserting {len(to_insert)} documents into database...")
        
        max_retries = 3
        retry_delay = 2
        inserted_count = 0
        for attempt in range(max_retries):
            try:
                result = config.collection.insert_many(to_insert, ordered=False)
                inserted_count = len(result.inserted_ids)
                print_log(f"[Task {task_id}] Successfully inserted {inserted_count} documents.")
                break
            except OperationFailure as e:
                if attempt < max_retries - 1:
                    error_str = str(e).lower()
                    # Retry on transient errors
                    if "duplicate" in error_str or "write concern" in error_str or "timeout" in error_str:
                        print_log(f"[Task {task_id}] Retry {attempt + 1}/{max_retries} after error: {e}")
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print_log(f"[Task {task_id}] Retry {attempt + 1}/{max_retries} after error: {e}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise
        
        # Step 6: Verify insertion and embedding presence
        tasks[task_id] = {"status": "processing", "step": "Verifying data..."}
        verification_queries = {}
        for model_name in all_embeddings.keys():
            if all_embeddings[model_name]:
                vec_field = EMBEDDING_CONFIG[model_name]["vector_field"]
                count = config.collection.count_documents({
                    f"metadata.{SESSION_FIELD}": session_id,
                    f"metadata.source": source,
                    vec_field: {"$exists": True}
                })
                verification_queries[model_name] = count
                print_log(f"[Task {task_id}] Verified: {count} documents with '{vec_field}' embeddings for '{model_name}'")
        
        if not any(verification_queries.values()):
            raise ValueError("No documents with embeddings found after insertion. Data may not have been saved correctly.")
        
        final_message = f"Successfully ingested {inserted_count} chunks from source '{source}' into session '{session_id}'. Embeddings: {', '.join([f'{k}: {v}' for k, v in verification_queries.items()])}"
        tasks[task_id] = {"status": "complete", "message": final_message}
        print_log(f"[Task {task_id}] ✅ {final_message}")
        print_log(f"[Task {task_id}] ⚠️  Note: Vector search indexes may take a few moments to index new documents. If search doesn't work immediately, wait 10-30 seconds and try again.")

    except ValueError as e:
        error_message = f"Ingestion validation failed: {str(e)}"
        print_log(f"[Task {task_id}] [ERROR] {error_message}")
        tasks[task_id] = {"status": "failed", "message": error_message}
    except OperationFailure as e:
        error_message = f"Database operation failed: {str(e)}"
        print_log(f"[Task {task_id}] [ERROR] {error_message}\n{traceback.format_exc()}")
        tasks[task_id] = {"status": "failed", "message": error_message}
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
    vector_field = model_config['vector_field']
    index_name = model_config['index_name']
    
    # Debug: Check if index is queryable
    index_info = _get_search_index(config.collection, index_name)
    if not index_info:
        print_log(f"[WARN] Index '{index_name}' not found. Vector search may fail.")
    elif not index_info.get("queryable", False):
        print_log(f"[WARN] Index '{index_name}' exists but is not queryable yet (Status: {index_info.get('status', 'UNKNOWN')}).")
    
    # Debug: Check if documents exist with embeddings for this session
    doc_count = config.collection.count_documents({
        f"metadata.{SESSION_FIELD}": session_id,
        vector_field: {"$exists": True}
    })
    print_log(f"[DEBUG] Found {doc_count} documents with '{vector_field}' embeddings in session '{session_id}'")
    
    if doc_count == 0:
        print_log(f"[WARN] No documents with embeddings found. Checking all documents in session...")
        total_docs = config.collection.count_documents({f"metadata.{SESSION_FIELD}": session_id})
        print_log(f"[DEBUG] Total documents in session '{session_id}': {total_docs}")
        if total_docs > 0:
            # Sample a document to see its structure
            sample = config.collection.find_one({f"metadata.{SESSION_FIELD}": session_id})
            if sample:
                has_embedding = vector_field in sample
                print_log(f"[DEBUG] Sample document has '{vector_field}': {has_embedding}")
                print_log(f"[DEBUG] Sample document keys: {list(sample.keys())}")
                if "metadata" in sample:
                    print_log(f"[DEBUG] Sample metadata keys: {list(sample.get('metadata', {}).keys())}")

    query_vector = embedding_client.embed_query(query)
    print_log(f"[DEBUG] Generated query vector of length {len(query_vector)} for model '{embedding_model}'")

    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": vector_field,
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

    try:
        results = list(config.collection.aggregate(pipeline))
        print_log(f"[DEBUG] Vector search returned {len(results)} results")
        return results
    except OperationFailure as e:
        error_details = str(e)
        print_log(f"[ERROR] Vector search failed: {error_details}")
        
        # If index not ready, try fallback: regular text search
        if "index" in error_details.lower() or "not found" in error_details.lower():
            print_log(f"[WARN] Attempting fallback text search...")
            # Fallback: simple text search
            results = list(config.collection.find(
                {
                    f"metadata.{SESSION_FIELD}": session_id,
                    "text": {"$regex": query, "$options": "i"}
                },
                {"_id": 0, "text": 1, "metadata.source": 1}
            ).limit(num_sources))
            
            # Add dummy scores for consistency
            for r in results:
                r["content"] = r.pop("text", "")
                r["source"] = r.get("metadata", {}).get("source", "N/A")
                r["score"] = 0.5  # Dummy score
                r.pop("metadata", None)
            
            print_log(f"[INFO] Fallback search returned {len(results)} results")
            return results
        
        raise


# --- 6. AGENT TOOLS ---
@tool
def search_knowledge_base(query: str, embedding_model: str, num_sources: int = 3, max_chunk_length: int = 2000) -> str:
    """Query the knowledge base to find relevant chunks for `query`."""
    try:
        print_log(f"[INFO] Searching with '{embedding_model}' → top {num_sources} in session '{config.current_session}'")
        results_with_scores = _perform_vector_search(query, config.current_session, embedding_model, num_sources)

        if not results_with_scores:
            config.last_retrieved_sources = []
            # Additional diagnostic: check if session has any data at all
            total_docs = config.collection.count_documents({f"metadata.{SESSION_FIELD}": config.current_session})
            if total_docs == 0:
                return f"No documents found in session '{config.current_session}'. Please add sources first."
            else:
                vector_field = EMBEDDING_CONFIG[embedding_model]["vector_field"]
                docs_with_embeddings = config.collection.count_documents({
                    f"metadata.{SESSION_FIELD}": config.current_session,
                    vector_field: {"$exists": True}
                })
                return f"No relevant info found. Session has {total_docs} docs, {docs_with_embeddings} with '{embedding_model}' embeddings. Index may still be building."

        # Remember sources
        found_sources = [r.get("source", "N/A") for r in results_with_scores]
        config.last_retrieved_sources = list(set(found_sources))
        
        # Debug: Store retrieved chunks
        import datetime
        config.debug_retrieved_chunks.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "session_id": config.current_session,
            "embedding_model": embedding_model,
            "chunks": [
                {
                    "text": r.get("content", ""),
                    "source": r.get("source", "N/A"),
                    "score": r.get("score", 0.0)
                }
                for r in results_with_scores
            ]
        })
        # Keep only last 50 retrievals
        if len(config.debug_retrieved_chunks) > 50:
            config.debug_retrieved_chunks = config.debug_retrieved_chunks[-50:]

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
        error_msg = f"Search error: {e}"
        print_log(f"[ERROR] search_knowledge_base: {error_msg}\n{traceback.format_exc()}")
        return f"❌ {error_msg}"

@tool
def read_url(url: str, chunk_size: int=1000, chunk_overlap: int=150) -> str:
    """Adds a URL's content (via Firecrawl HTTP API) into the knowledge base."""
    try:
        if config.collection.find_one({"metadata.source": url, f"metadata.{SESSION_FIELD}": config.current_session}):
            return f"❌ Source '{url}' already exists in session '{config.current_session}'."

        if not config.firecrawl_api_key:
            firecrawl_key = os.getenv("FIRECRAWL_API_KEY", "NOT_SET")
            return f"❌ FIRECRAWL_API_KEY not set. (Key present: {firecrawl_key != 'NOT_SET'})"

        print_log(f"[INFO] Scrape & Ingest URL via Firecrawl HTTP API: {url}")
        
        # Use Firecrawl HTTP API directly
        api_url = "https://api.firecrawl.dev/v2/scrape"
        headers = {
            "Authorization": f"Bearer {config.firecrawl_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": False
        }
        
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        scrape_result = response.json()
        
        # Extract markdown content from response
        # Firecrawl API response structure: {"data": {"markdown": "..."}}
        page_content = ""
        if "data" in scrape_result:
            if isinstance(scrape_result["data"], dict):
                page_content = scrape_result["data"].get("markdown", "")
            elif isinstance(scrape_result["data"], str):
                # Sometimes the API returns data as a string directly
                page_content = scrape_result["data"]
        elif "markdown" in scrape_result:
            page_content = scrape_result["markdown"]
        
        if not page_content:
            print_log(f"[WARN] Firecrawl response structure: {list(scrape_result.keys())}")
            return f"❌ No markdown content returned from {url}. Response: {str(scrape_result)[:200]}"

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(page_content)
        if not chunks:
            return "❌ Could not split content into chunks."

        all_embeddings = _embed_chunks_parallel(chunks)
        if not any(emb_list for emb_list in all_embeddings.values()):
            return "❌ Failed to generate embeddings for any model."

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
                if embed_list and i < len(embed_list):
                    vector_field = EMBEDDING_CONFIG[model_name]["vector_field"]
                    doc[vector_field] = embed_list[i]
            docs_to_insert.append(doc)

        # Insert with retry logic for robustness
        max_retries = 3
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                config.collection.insert_many(docs_to_insert, ordered=False)
                break
            except OperationFailure as e:
                if attempt < max_retries - 1:
                    error_str = str(e).lower()
                    if "duplicate" in error_str or "write concern" in error_str or "timeout" in error_str:
                        print_log(f"[WARN] Retry {attempt + 1}/{max_retries} for URL ingestion: {e}")
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                raise
        
        return f"✅ Ingested {len(chunks)} chunks from {url} into '{config.current_session}'."

    except requests.exceptions.RequestException as e:
        print_log(f"[ERROR] read_url HTTP error: {e}\n{traceback.format_exc()}")
        return f"❌ HTTP error fetching URL: {e}"
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
        res = _delete_chunk_from_db(chunk_id)
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


# Create LangGraph agent (modern replacement for AgentExecutor)
agent_executor = create_react_agent(llm, tools)


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

        # Prepare messages for LangGraph agent
        # LangGraph uses a messages list format instead of input string
        # Include system prompt in the first message if no history exists
        messages = []
        if not current_chat_history:
            # Add system prompt as the first message when starting fresh
            messages.append(SystemMessage(content=AGENT_SYSTEM_PROMPT))
        
        # Add chat history
        messages.extend(list(current_chat_history))
        
        # Add system instruction and user query
        system_instruction = (
            f"IMPORTANT INSTRUCTION: When you call the 'search_knowledge_base' tool, "
            f"you MUST set the 'embedding_model' parameter to '{embedding_model}'."
        )
        messages.append(HumanMessage(content=f"{system_instruction}\n\nUser query: '{user_input}'"))

        # Agent call with LangGraph format
        response = agent_executor.invoke({
            "messages": messages
        })

        # Extract the final answer from LangGraph response
        # LangGraph returns the full state with messages array
        final_messages = response.get("messages", [])
        if final_messages:
            # The last message is the AI's response
            final_answer = final_messages[-1].content
        else:
            final_answer = "I apologize, but I couldn't generate a response."

        # Record the conversation
        current_chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=final_answer)
        ])
        config.chat_history[session_id] = current_chat_history
        
        sources_used = config.last_retrieved_sources
        
        # Debug: Store request info
        import datetime
        config.debug_requests.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "query": user_input,
            "session_id": session_id,
            "embedding_model": embedding_model,
            "num_sources": num_sources,
            "sources_used": sources_used,
            "response_length": len(final_answer),
            "rag_params": rag_params
        })
        # Keep only last 50 requests
        if len(config.debug_requests) > 50:
            config.debug_requests = config.debug_requests[-50:]

        messages = [{
            "type": "bot-message",
            "content": final_answer,
            "sources": sources_used,
            "query": user_input  # Include query for chunk inspection
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
    
    session_id = request.args.get("session_id", config.current_session)
    
    # Get index readiness info for the session
    index_status = {}
    for model_name, model_cfg in EMBEDDING_CONFIG.items():
        if model_name not in config.embedding_clients:
            continue
        
        vector_field = model_cfg["vector_field"]
        index_name = model_cfg["index_name"]
        
        # Count documents with embeddings
        doc_count = config.collection.count_documents({
            f"metadata.{SESSION_FIELD}": session_id,
            vector_field: {"$exists": True}
        })
        
        # Check index status
        index_info = _get_search_index(config.collection, index_name)
        index_queryable = index_info.get("queryable", False) if index_info else False
        index_status_str = index_info.get("status", "UNKNOWN") if index_info else "NOT_FOUND"
        
        # Auto-create index if missing and we have documents (background task)
        if index_status_str == "NOT_FOUND" and doc_count > 0:
            # Check if we're already creating this index
            if index_name not in index_creation_in_progress:
                print_log(f"[INFO] Index '{index_name}' missing but {doc_count} documents exist. Triggering auto-creation...")
                index_creation_in_progress.add(index_name)
                
                # Trigger index creation in background
                def create_index_background():
                    try:
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
                        print_log(f"[INFO] Creating index '{index_name}' with {doc_count} documents...")
                        _create_or_update_search_index(
                            collection=config.collection,
                            index_name=index_name,
                            definition=definition,
                            wait_for_ready=False,  # Don't wait, but creation will happen
                            timeout=600
                        )
                        print_log(f"[INFO] Index '{index_name}' creation submitted successfully.")
                    except Exception as e:
                        print_log(f"[ERROR] Background index creation failed for '{index_name}': {e}\n{traceback.format_exc()}")
                        index_creation_in_progress.discard(index_name)
                    finally:
                        # Remove from in-progress after a delay to allow status checks
                        # The index status will update once it's actually created
                        pass
                
                executor.submit(create_index_background)
            
            # Re-check index status - it might have been created by another request
            index_info = _get_search_index(config.collection, index_name)
            if index_info:
                index_queryable = index_info.get("queryable", False)
                index_status_str = index_info.get("status", "BUILDING")
            else:
                index_status_str = "CREATING"  # Update status to reflect creation in progress
        elif index_name in index_creation_in_progress:
            # Index creation is in progress, check if it's done
            index_info = _get_search_index(config.collection, index_name)
            if index_info:
                index_queryable = index_info.get("queryable", False)
                index_status_str = index_info.get("status", "BUILDING")
                if index_queryable:
                    # Index is ready, remove from in-progress
                    index_creation_in_progress.discard(index_name)
            else:
                index_status_str = "CREATING"
        
        index_status[model_name] = {
            "document_count": doc_count,
            "index_queryable": index_queryable,
            "index_status": index_status_str,
            "index_ready": index_queryable and doc_count > 0
        }

    return jsonify({
        "all_sessions": sorted(list(all_sessions)),
        "current_session": config.current_session,
        "available_embedding_models": list(config.embedding_clients.keys()),
        "index_status": index_status
    })

@app.route("/index_status", methods=["GET"])
def get_index_status():
    """Get detailed index status for a session."""
    session_id = request.args.get("session_id", config.current_session)
    embedding_model = request.args.get("embedding_model", "openai")
    auto_create = request.args.get("auto_create", "false").lower() == "true"
    
    if embedding_model not in config.embedding_clients:
        return jsonify({"error": f"Embedding model '{embedding_model}' not available"}), 400
    
    model_cfg = EMBEDDING_CONFIG[embedding_model]
    vector_field = model_cfg["vector_field"]
    index_name = model_cfg["index_name"]
    dims = model_cfg["dimensions"]
    
    # Count documents with embeddings
    doc_count = config.collection.count_documents({
        f"metadata.{SESSION_FIELD}": session_id,
        vector_field: {"$exists": True}
    })
    
    # Check index status
    index_info = _get_search_index(config.collection, index_name)
    index_queryable = index_info.get("queryable", False) if index_info else False
    index_status_str = index_info.get("status", "UNKNOWN") if index_info else "NOT_FOUND"
    
    # Auto-create index if missing and we have documents
    if index_status_str == "NOT_FOUND" and doc_count > 0 and auto_create:
        if index_name not in index_creation_in_progress:
            print_log(f"[INFO] Index '{index_name}' not found but {doc_count} documents exist. Creating index...")
            index_creation_in_progress.add(index_name)
            try:
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
                _create_or_update_search_index(
                    collection=config.collection,
                    index_name=index_name,
                    definition=definition,
                    wait_for_ready=False,  # Don't wait, return immediately
                    timeout=600
                )
                print_log(f"[INFO] Index '{index_name}' creation submitted.")
            except Exception as e:
                print_log(f"[ERROR] Failed to auto-create index '{index_name}': {e}\n{traceback.format_exc()}")
                index_creation_in_progress.discard(index_name)
                index_status_str = "CREATION_FAILED"
        
        # Re-check status after creation attempt
        index_info = _get_search_index(config.collection, index_name)
        if index_info:
            index_queryable = index_info.get("queryable", False)
            index_status_str = index_info.get("status", "BUILDING")
            if index_queryable:
                index_creation_in_progress.discard(index_name)
        else:
            index_status_str = "CREATING"
    elif index_name in index_creation_in_progress:
        # Index creation is in progress, check if it's done
        index_info = _get_search_index(config.collection, index_name)
        if index_info:
            index_queryable = index_info.get("queryable", False)
            index_status_str = index_info.get("status", "BUILDING")
            if index_queryable:
                index_creation_in_progress.discard(index_name)
        else:
            index_status_str = "CREATING"
    
    return jsonify({
        "session_id": session_id,
        "embedding_model": embedding_model,
        "document_count": doc_count,
        "index_name": index_name,
        "index_queryable": index_queryable,
        "index_status": index_status_str,
        "index_ready": index_queryable and doc_count > 0,
        "ready_for_search": index_queryable and doc_count > 0
    })

@app.route("/indexes/create", methods=["POST"])
def create_indexes():
    """Create or update all vector search indexes."""
    try:
        print_log("[INFO] Manual index creation requested...")
        setup_database_and_index()
        return jsonify({"status": "success", "message": "Index creation initiated. Check index status for progress."})
    except Exception as e:
        print_log(f"[ERROR] Failed to create indexes: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/debug", methods=["GET"])
def get_debug_info():
    """Get debug information including chunks, LLM requests, and system state."""
    session_id = request.args.get("session_id", config.current_session)
    embedding_model = request.args.get("embedding_model", "openai")
    
    # Get recent requests for this session
    session_requests = [
        req for req in config.debug_requests 
        if req.get("session_id") == session_id
    ][-10:]  # Last 10 requests
    
    # Get recent retrieved chunks for this session
    session_chunks = [
        chunk for chunk in config.debug_retrieved_chunks
        if chunk.get("session_id") == session_id and chunk.get("embedding_model") == embedding_model
    ][-10:]  # Last 10 retrievals
    
    # Get index status
    model_cfg = EMBEDDING_CONFIG.get(embedding_model, {})
    index_name = model_cfg.get("index_name", "")
    vector_field = model_cfg.get("vector_field", "")
    
    doc_count = config.collection.count_documents({
        f"metadata.{SESSION_FIELD}": session_id,
        vector_field: {"$exists": True}
    }) if vector_field else 0
    
    index_info = _get_search_index(config.collection, index_name) if index_name else None
    
    # Get sample chunks from database
    sample_chunks = list(config.collection.find(
        {f"metadata.{SESSION_FIELD}": session_id},
        {"text": 1, "metadata.source": 1, "_id": 1}
    ).limit(20))
    
    return jsonify({
        "session_id": session_id,
        "embedding_model": embedding_model,
        "index_status": {
            "name": index_name,
            "status": index_info.get("status", "UNKNOWN") if index_info else "NOT_FOUND",
            "queryable": index_info.get("queryable", False) if index_info else False,
            "document_count": doc_count
        },
        "recent_requests": session_requests,
        "recent_retrieved_chunks": session_chunks,
        "sample_chunks": [
            {
                "_id": str(chunk["_id"]),
                "text": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", ""),
                "source": chunk.get("metadata", {}).get("source", "N/A")
            }
            for chunk in sample_chunks
        ],
        "chat_history_length": len(config.chat_history.get(session_id, [])),
        "total_requests_stored": len(config.debug_requests),
        "total_chunks_stored": len(config.debug_retrieved_chunks)
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
    
    # --- CHANGED: Using Firecrawl HTTP API directly ---
    if not config.firecrawl_api_key:
        firecrawl_key = os.getenv("FIRECRAWL_API_KEY", "NOT_SET")
        error_msg = f"FIRECRAWL_API_KEY not set. (Key present in env: {firecrawl_key != 'NOT_SET'}, Key length: {len(firecrawl_key) if firecrawl_key != 'NOT_SET' else 0})"
        print_log(f"[ERROR] {error_msg}")
        return jsonify({"error": error_msg}), 500

    try:
        print_log(f"[INFO] Previewing URL via Firecrawl HTTP API: {url}")
        
        # Use Firecrawl HTTP API directly
        api_url = "https://api.firecrawl.dev/v2/scrape"
        headers = {
            "Authorization": f"Bearer {config.firecrawl_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": False
        }
        
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        scrape_result = response.json()
        
        # Extract markdown content from response
        page_content = ""
        if "data" in scrape_result:
            if isinstance(scrape_result["data"], dict):
                page_content = scrape_result["data"].get("markdown", "")
            elif isinstance(scrape_result["data"], str):
                page_content = scrape_result["data"]
        elif "markdown" in scrape_result:
            page_content = scrape_result["markdown"]

        MAX_PREVIEW = 50000
        if len(page_content) > MAX_PREVIEW:
            page_content = page_content[:MAX_PREVIEW] + "\n\n[TRUNCATED]"
        return jsonify({"markdown": page_content})

    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching URL content via Firecrawl HTTP API: {e}"
        print_log(f"[ERROR] {error_msg}")
        return jsonify({"error": error_msg}), 500
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print_log(f"[ERROR] {error_msg}\n{traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500

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
def _wait_for_search_index_ready(collection, index_name: str, timeout: int = 600, poll_interval: int = 5) -> bool:
    """
    Polls the index status until it becomes queryable or fails.
    Returns True if index is ready, False if failed, raises TimeoutError on timeout.
    """
    start_time = time.time()
    print_log(f"[INFO] Waiting up to {timeout}s for search index '{index_name}' to become queryable...")
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            error_msg = f"Timeout: Index '{index_name}' did not become queryable within {timeout}s."
            print_log(f"[ERROR] {error_msg}")
            raise TimeoutError(error_msg)
        
        try:
            # Poll for the index status
            existing = next(collection.list_search_indexes(name=index_name), None)
            if existing:
                status = existing.get("status", "UNKNOWN")
                queryable = existing.get("queryable", False)
                
                if status == "FAILED":
                    error_msg = f"Search index '{index_name}' failed to build (Status: FAILED). Check Atlas UI for details."
                    print_log(f"[ERROR] {error_msg}")
                    raise Exception(error_msg)
                
                if queryable:
                    print_log(f"[INFO] Search index '{index_name}' is queryable (Status: {status}).")
                    return True
                
                # Not ready yet, log and wait
                print_log(f"[INFO] Polling for '{index_name}'. Status: {status}. Queryable: {queryable}. Elapsed: {elapsed:.0f}s")
            else:
                # Index not found yet (can happen right after creation command)
                print_log(f"[INFO] Polling for '{index_name}'. Index not found yet (normal during creation). Elapsed: {elapsed:.0f}s")
        
        except (OperationFailure, ConnectionFailure) as e:
            # Handle transient network/DB errors during polling
            print_log(f"[WARN] DB Error during polling for index '{index_name}': {e}. Retrying...")
        except Exception as e:
            print_log(f"[WARN] Unexpected error during polling for index '{index_name}': {e}. Retrying...")
        
        time.sleep(poll_interval)

def _get_search_index(collection, index_name: str):
    """Retrieves a search index by name, returns None if not found."""
    try:
        return next(collection.list_search_indexes(name=index_name), None)
    except Exception as e:
        print_log(f"[WARN] Error retrieving search index '{index_name}': {e}")
        return None

def _ensure_collection_exists(db, collection_name: str):
    """Ensures the collection exists, creating it if necessary."""
    try:
        if collection_name not in db.list_collection_names():
            db.create_collection(collection_name)
            print_log(f"[INFO] Created collection '{collection_name}'.")
        else:
            print_log(f"[INFO] Collection '{collection_name}' already exists.")
    except OperationFailure as e:
        if "already exists" in str(e).lower() or "CollectionInvalid" in str(type(e).__name__):
            print_log(f"[INFO] Collection '{collection_name}' already exists (race condition).")
        else:
            print_log(f"[WARN] Error ensuring collection '{collection_name}' exists: {e}")
            raise
    except Exception as e:
        print_log(f"[WARN] Unexpected error ensuring collection '{collection_name}' exists: {e}")
        raise

def _create_or_update_search_index(collection, index_name: str, definition: dict, wait_for_ready: bool = True, timeout: int = 600) -> bool:
    """
    Creates or updates a search index with proper error handling and race condition management.
    Returns True if index is ready/queryable, False otherwise.
    """
    try:
        # Check for existing index
        existing_index = _get_search_index(collection, index_name)
        
        if existing_index:
            print_log(f"[INFO] Search index '{index_name}' already exists.")
            latest_def = existing_index.get("latestDefinition", {})
            definition_changed = False
            change_reason = ""
            
            # Compare definition for changes
            if "fields" in definition:
                existing_fields = latest_def.get("fields")
                if existing_fields != definition["fields"]:
                    definition_changed = True
                    change_reason = "vector 'fields' definition differs."
            
            if definition_changed:
                print_log(f"[WARN] Search index '{index_name}' definition has changed ({change_reason}). Triggering update...")
                try:
                    collection.update_search_index(name=index_name, definition=definition)
                    print_log(f"[INFO] Search index '{index_name}' update submitted.")
                except OperationFailure as e:
                    if "IndexNotFound" in str(e):
                        print_log(f"[WARN] Index '{index_name}' was deleted during update. Will recreate.")
                        definition_changed = True  # Force recreation
                    else:
                        print_log(f"[ERROR] Failed to update index '{index_name}': {e}")
                        raise
            
            if not definition_changed and existing_index.get("queryable"):
                print_log(f"[INFO] Search index '{index_name}' is already queryable and up-to-date.")
                return True
            elif existing_index.get("status") == "FAILED":
                print_log(f"[ERROR] Search index '{index_name}' exists but is in FAILED state. Manual intervention may be required.")
                return False
            elif not definition_changed:
                # Index exists, is up-to-date, but not queryable yet
                print_log(f"[INFO] Search index '{index_name}' exists and is up-to-date, but not queryable (Status: {existing_index.get('status')}). Waiting...")
        else:
            # Create new index
            try:
                print_log(f"[ACTION] Creating new search index '{index_name}'...")
                search_index_model = SearchIndexModel(
                    definition=definition,
                    name=index_name,
                    type="vectorSearch"
                )
                collection.create_search_index(model=search_index_model)
                print_log(f"[INFO] Search index '{index_name}' build has been submitted.")
            except OperationFailure as e:
                # Handle race condition where another process created the index
                if "IndexAlreadyExists" in str(e) or "DuplicateIndexName" in str(e) or "already exists" in str(e).lower():
                    print_log(f"[WARN] Race condition: Index '{index_name}' was created by another process.")
                else:
                    print_log(f"[ERROR] OperationFailure during search index creation for '{index_name}': {e}")
                    raise
        
        # Wait for ready if requested
        if wait_for_ready:
            return _wait_for_search_index_ready(collection, index_name, timeout)
        return True
    
    except OperationFailure as e:
        print_log(f"[ERROR] OperationFailure during search index creation/check for '{index_name}': {e}")
        raise
    except Exception as e:
        print_log(f"[ERROR] Unexpected error regarding search index '{index_name}': {e}")
        raise

def setup_database_and_index():
    """Initializes database, collection, and vector search indexes with robust error handling."""
    print_log("--- 🚀 Initializing DB and Vector Search Indexes ---")
    
    # Ensure collection exists first
    try:
        _ensure_collection_exists(config.db, COLLECTION_NAME)
    except Exception as e:
        print_log(f"[FATAL] Failed to ensure collection exists: {e}")
        raise
    
    # Create indexes for each available embedding model
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
            _create_or_update_search_index(
                collection=config.collection,
                index_name=index_name,
                definition=definition,
                wait_for_ready=True,
                timeout=600  # 10 minutes timeout
            )
            print_log(f"[INFO] ✅ Index '{index_name}' for model '{model_name}' is ready.")
        except TimeoutError:
            print_log(f"[ERROR] ⚠️ Index '{index_name}' creation timed out. It may still be building in the background.")
            # Don't raise - allow app to start, index will be ready later
        except Exception as e:
            print_log(f"[ERROR] ⚠️ Failed to create index '{index_name}': {e}. App will continue, but vector search may not work until index is created.")
            # Don't raise - allow app to start
    
    print_log("--- ✅ Database and index initialization complete ---")

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