# A Unified Approach to Chunking and RAG Agents: Building a Smarter AI System  
  
The recent surge of Large Language Models (LLMs) has fundamentally changed how we interact with information. However, summarizing and querying lengthy documents still poses challenges. Two key strategies—**chunking** and **Retrieval-Augmented Generation (RAG)**—are at the core of solving these issues. In addition, a **unified architecture** that keeps content, metadata, and vector representations together can streamline data management and enhance AI capabilities. Below, we present two complementary perspectives on creating a smarter RAG agent, each focusing on how MongoDB Atlas Vector Search and a document-based approach can simplify and strengthen your system.  
  
---  
  
## Part 1: Chunking and RAG Agents—Building a Smarter, Unified AI System  
  
### <h2>Chunking and RAG Agents: Building a Smarter, Unified AI System</h2>  
  
The recent boom in Large Language Models (LLMs) has revolutionized how we interact with and understand information. Yet, effectively summarizing and querying long documents still presents challenges. This is where **chunking**, the process of dividing documents into smaller, meaningful units, and **Retrieval-Augmented Generation (RAG)**, a two-pronged approach that combines information retrieval with text generation, become crucial. When powered by a unified data architecture, such as the one offered by **MongoDB Atlas Vector Search**, these concepts can create a flexible and powerful AI agent.  
  
-----  
  
### <h3>The Problem with Fragmented RAG Architectures</h3>  
  
A typical RAG system often operates like a fragmented mess. Raw documents reside in one system, their extracted text and embeddings are stored in a separate vector database, and the associated metadata lives somewhere else entirely. This siloed approach complicates data management, updates, and experimentation.  
  
A better way exists. By leveraging the flexible document model of a modern database like **MongoDB**, you can keep all three components—the content, its metadata, and its vector representation—in a single place.  
  
Consider a small piece of text from a website you've ingested. You can store it as a single JSON document:  
  
```json  
{  
  "_id": ObjectId("..."),  
  "text": "MongoDB's document model stores data as BSON documents...",  
  "metadata": {  
    "source": "https://www.mongodb.com/docs/",  
    "source_type": "url",  
    "session_id": "product_faq"  
  },  
  "embedding_openai": [0.123, 0.456, ...],  
  "embedding_voyageai": [0.789, 0.101, ...]  
}  
```  
  
This simple structure solves several key problems. First, its **schema-agnostic document model** allows you to store vectors from multiple different embedding models in the same document. This means you can easily A/B test a new model without having to migrate your data. Second, you can perform a vector search while simultaneously filtering by metadata, instantly making your search results more relevant and scoped. This unified approach simplifies data management and retrieval by bringing your data, its context, and its vector representation together.  
  
-----  
  
### <h3>The Art of Chunking: A Critical Decision</h3>  
  
The effectiveness of any RAG system hinges on a well-chunked knowledge base. A "chunk" is a small, semantically meaningful unit of text that the system can use for searching and summarization.  
  
Researchers are actively exploring various chunking strategies to optimize RAG performance:  
  
* **Fixed-size chunks with overlap:** This method divides the document into chunks of a predetermined size, with a character overlap between them to maintain context. A good starting point is a chunk size of **1,000 characters** with an overlap of **150 characters**. This ensures sentences or important phrases aren't cut in half.    
* **Recursive chunking:** This strategy takes an iterative approach, starting with the entire document and then splitting each chunk into smaller and smaller pieces. This allows for fine-grained control and is particularly useful for complex, nested documents.    
* **Paragraph-based chunking:** This method uses natural paragraph breaks to define chunk boundaries, making it suitable for documents with well-defined paragraphs.    
* **Single-page chunks:** While simple and efficient, this approach may not capture crucial details or address the limitations of LLM processing. It can be enhanced with hybrid search to retrieve relevant information based on both text and visual cues.  
  
By leveraging **LangChain's `RecursiveCharacterTextSplitter`**, we can intelligently split documents while preserving semantic meaning, which is superior to fixed-size splitting. This intelligent chunking, combined with MongoDB Atlas Vector Search, allows for efficient Approximate Nearest Neighbor (ANN) searches using the `$vectorSearch` operator, ensuring the most relevant passages are retrieved for summarization.  
  
-----  
  
### <h3>Retrieval Parameters: Tuning for Precision</h3>  
  
Once your data is chunked and ingested, the next challenge is retrieval. Our system gives the user control over key retrieval parameters to find the most relevant chunks.  
  
#### <h4>The `min_rel_score`: A Quality Check</h4>  
  
The `min_rel_score` (minimum relevance score) acts as a critical quality filter. Vector search returns results ranked by how similar their embedding is to the query's embedding, with a score between 0 and 1.  
  
* A **high `min_rel_score`** (e.g., 0.80) is useful for specific, high-quality searches. If a user asks a vague question and all returned documents have a score of less than 0.70, the agent won't use them. This prevents the LLM from trying to answer with irrelevant or low-quality information, leading to a more reliable "I don't know" response.    
* A **low `min_rel_score`** (e.g., 0.20) is useful when you want to retrieve anything remotely related to the query, which can be helpful for exploratory searches.  
  
By setting this parameter, you can instruct the agent to be selective about what information it uses, preventing "garbage-in, garbage-out" scenarios.  
  
#### <h4>`num_sources` (k): Context is King</h4>  
  
The `num_sources` parameter, often referred to as 'k' in RAG literature, determines how many top-ranking chunks the agent should retrieve.  
  
* A **small k** (e.g., 3) is ideal for specific, factual questions. Too many chunks could introduce noise and confuse the LLM.    
* A **large k** (e.g., 10) is better for open-ended or complex queries that require a broader understanding, such as "Compare the pros and cons of using MongoDB Atlas vs. a self-hosted solution."  
  
This tunable parameter allows you to balance the need for concise, focused answers with the need for rich, comprehensive ones.  
  
-----  
  
### <h3>Beyond Retrieval: The Agent's Intelligence</h3>  
  
An intelligent RAG agent is not just about retrieving information; it's about actively managing its knowledge base. With a unified data architecture, this becomes intuitive. For instance, if a user asks about a policy that has recently changed, the agent could ask, "I'm still seeing the old policy. Would you like to update the knowledge base?"  
  
The user could then respond, and the agent, using a tool, could execute a command like:  
```  
update_chunk(chunk_id='ObjectId("...")', new_content='The new policy takes effect on Jan 1, 2026.')  
```  
This level of control is possible because each chunk has a unique ID, and our data store supports standard `update` operations on that document. This is a crucial, often missing, piece in RAG applications for maintaining the quality of a living knowledge base.  
  
-----  
  
### <h3>Under the Hood: The Aggregation Pipeline</h3>  
  
The power of this unified architecture is best seen in the **MongoDB Atlas aggregation pipeline**. This pipeline allows for a complex, multi-stage retrieval process that combines vector search with traditional filtering and data manipulation.  
  
```python  
agg_pipeline = [  
    {  
        "$vectorSearch": {  
            "index": 'nested_search_index',  
            "path": "text_embedding",  
            "queryVector": query_vector,  
            "limit": k,  
            "numCandidates": k * multiplier,  
        },  
    },  
    {  
        "$match": {"sample_question": {"$exists": False}}  
    },  
    {  
        "$project": {"text_embedding": 0}  
    },  
    {  
        '$lookup': {  
            "from": "hnsw_parent_retrieval_example",  
            "localField": "parent_id",  
            "foreignField": "_id",  
            "as": 'parent_documents'  
        }  
    },  
    {  
        '$unwind': {"path": "$parent_documents"}  
    },  
    {  
        "$limit": k  
    }  
]  
```  
  
This pipeline first uses `$vectorSearch` to find relevant documents based on semantic similarity. It then uses `$match` to filter out irrelevant documents, `$project` to remove unnecessary fields, and `$lookup` and `$unwind` to retrieve and merge information from parent documents. Finally, it uses `$limit` to return a refined set of the top **k** most relevant results.  
  
This is not just about summarizing text; it's about unlocking deeper understanding and transforming information into meaningful insights. By combining a question-driven approach with a unified data architecture, we open a new chapter in the field of text analysis and summarization, paving the way for a more insightful and impactful future.  
  
---  
  
## Part 2: Unlocking Flexibility and Control 
  
### <h2>Unlocking Flexibility and Control</h2>  
  
Agents are revolutionizing the way we interact with language models, transforming them into dynamic decision-making and task-performing systems. These reasoning engines are becoming vital for automating tasks, processing information, and improving human-computer interactions. One of the most powerful applications of agents is in **Retrieval Augmented Generation (RAG)**, where they can be used to build flexible and interactive systems.  
  
This project explores how you can build a more intelligent RAG agent by leveraging a unified architecture with MongoDB Atlas. This approach allows you to manage your data, metadata, and vectors in a single place, simplifying the entire RAG pipeline and enabling real-time conversational control over your RAG strategy.  
  
### <h3>The Problem with Fragmented RAG Architectures</h3>  
  
A typical RAG system is a fragmented mess. Your raw documents are in one place, their text is extracted and stored as embeddings in a separate vector database, and the metadata lives somewhere else entirely. This leads to a complex, multi-siloed architecture that makes data management, updates, and experimentation a nightmare.  
  
What if there was a better way? What if you could simplify this by keeping all three components—the content, its metadata, and its vector representation—in one place?  
  
### <h3>The Cohesion of a Single Document</h3>  
  
The core of our approach is leveraging the flexible document model of a modern database like **MongoDB**. Instead of splitting your data across different systems, you can treat each "chunk" of your knowledge as a single, self-contained entity.  
  
Consider a small piece of text from a website you've ingested. You can store it as a single JSON document:  
  
```json  
{  
  "_id": ObjectId("..."),  
  "text": "MongoDB's document model stores data as BSON documents...",  
  "metadata": {  
    "source": "https://www.mongodb.com/docs/",  
    "source_type": "url",  
    "session_id": "product_faq"  
  },  
  "embedding_openai": [0.123, 0.456, ...],  
  "embedding_voyageai": [0.789, 0.101, ...]  
}  
```  
  
This simple structure solves several key problems:  
  
* **Effortless Experimentation:** The schema-agnostic document model allows you to store vectors from multiple different embedding models in the same document. This means you can easily A/B test a new model without having to migrate your data.    
* **Precise Contextual Filtering:** You can perform a vector search while simultaneously filtering by metadata. For example, you can tell the database to only search for documents where `metadata.session_id == "product_faq"`, instantly making your search results more relevant and scoped.  
  
This unified approach brings your data, its context, and its vector representation together, simplifying data management and retrieval.  
  
-----  
  
### <h3>The Art of Chunking: A Critical Decision</h3>  
  
The effectiveness of any RAG system depends on a well-chunked knowledge base. A "chunk" is a small, semantically meaningful unit of text that the system can use for search.  
  
Our ingestion process uses **LangChain's `RecursiveCharacterTextSplitter`** to intelligently split documents. This method is superior to fixed-size splitting because it attempts to maintain sentence and paragraph boundaries, which is crucial for preserving the semantic meaning of each chunk.  
  
A key decision in this process is the **chunk size** and **chunk overlap**. Our application allows users to configure these parameters, which have a direct impact on the quality of the generated response.  
  
* **Chunk size:** The maximum number of characters in a chunk. A size of **1000 characters** is a good starting point.    
* **Chunk overlap:** The number of characters that repeat between chunks. An overlap of **150 characters** ensures that sentences or important phrases aren't cut in half at the boundary, providing better context for the LLM.  
  
Choosing the right chunking strategy is a fine art, balancing the need for small, focused chunks for accurate retrieval against the need for enough context to be meaningful. This is another area where our application's flexibility proves valuable, allowing us to easily adjust the chunking settings for different types of documents.  
  
-----  
  
### <h3>Retrieval Parameters: Tuning for Precision</h3>  
  
Once your data is chunked and ingested, the next challenge is retrieval. How do you find the most relevant chunks? It's not just a simple search; it's a conversation with the database. Our system gives the user control over key retrieval parameters right from the UI.  
  
#### <h4>The `min_rel_score`: A Quality Check</h4>  
  
The `min_rel_score` (minimum relevance score) is a critical parameter that acts as a quality filter. Vector search returns results ranked by how similar their embedding is to the query's embedding, with a score between 0 and 1.  
  
* **High `min_rel_score` (e.g., 0.80):** This is useful for specific, high-quality searches. If a user asks a vague question and all returned documents have a score of less than 0.70, the agent won't use them. This prevents the LLM from trying to answer with irrelevant or low-quality information, leading to a more reliable "I don't know" response.    
* **Low `min_rel_score` (e.g., 0.20):** A low threshold is useful when you want to retrieve anything remotely related to the query, which can be helpful for exploratory searches.  
  
By setting the `min_rel_score`, you can instruct the agent to be selective about what information it uses. This prevents "garbage-in, garbage-out" scenarios and is a powerful tool for maintaining response quality.  
  
#### <h4>`num_sources` (k): Context is King</h4>  
  
The `num_sources` parameter, often referred to as 'k' in RAG literature, determines how many top-ranking chunks the agent should retrieve.  
  
* **Small k (e.g., 3):** Ideal for specific, factual questions. Too many chunks could introduce noise and confuse the LLM.    
* **Large k (e.g., 10):** Better for open-ended or complex queries that require a broader understanding, such as "Compare the pros and cons of using MongoDB Atlas vs. a self-hosted solution."  
  
This tunable parameter allows you to balance the need for concise, focused answers with the need for rich, comprehensive ones.  
  
  
-----  
  
### <h3>Appendix: Under the Hood</h3>  
  
#### <h4>Tools as the Agent's Hands</h4>  
  
In this application, we use **LangChain's** tool-calling capabilities to allow the agent to interact with our backend. Each tool is a Python function decorated with `@tool`, which includes a docstring that tells the agent what the tool does. The LLM's job is to select and execute the right tool for the user's request.  
  
For example, here's a simplified version of our `search_knowledge_base` tool:  
  
```python  
@tool  
def search_knowledge_base(query: str, embedding_model: str, num_sources: int = 3) -> str:  
    """Searches the knowledge base for relevant chunks given a query."""  
    # This function uses a vector search pipeline to retrieve documents  
    # The pipeline is passed the user's query vector, embedding model, and number of sources  
  
    pipeline = [  
        {  
            "$vectorSearch": {  
                "index": "vector_index",  
                "path": embedding_model_field, # 'embedding_openai' or 'embedding_voyageai'  
                "queryVector": query_vector,  
                "numCandidates": num_sources * 10,  
                "limit": num_sources,  
                "filter": {  
                    "metadata.session_id": {"$eq": session_id} # Contextual filter  
                }  
            }  
        },  
        # ... rest of the pipeline to project the results  
    ]  
  
    # Execute aggregation pipeline on MongoDB collection  
    results = list(config.collection.aggregate(pipeline))  
    return format_results(results)  
```  
  
The LLM intelligently parses the user's request, pulls out arguments like `num_sources`, and calls this function. This architecture keeps the core LLM focused on its primary task—reasoning and language generation—while delegating complex database interactions to purpose-built tools.  
  
---  
  
## Conclusion  
  
By bringing chunking, RAG, and a unified data architecture under one roof, you can build AI agents that are both powerful and flexible. Throughout these two perspectives, we’ve seen how MongoDB’s document model simplifies data management, how intelligent chunking strategies enhance retrieval, and how careful tuning of relevance scores and source counts refines results. Most importantly, by treating each “chunk” as a self-contained entity in a single database, updates and experimentation become frictionless—paving the way for a truly dynamic, living knowledge base.  
  
Whether you’re aiming for more accurate question answering, deeper policy management, or ongoing experimentation with multiple embeddings, a unified RAG architecture redefines what’s possible in text analysis and summarization. We hope these insights help you build an even smarter, more controllable RAG agent for your own applications.