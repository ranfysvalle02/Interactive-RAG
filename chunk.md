## Chunking: A Hidden Hero in the Rise of GenAI

## ![Alt text](https://cdn.stackoverflow.co/images/jo7n4k8s/production/ef172115fca9aa6b3b99eeb1c749acf9f8c183a0-6000x3150.png?w=1200&h=630&auto=format&dpr=2)

The recent boom in Large Language Models (LLMs) has opened up a new world of possibilities for understanding and interacting with language. One of the most exciting applications is their ability to automatically summarize long documents, saving us valuable time and effort. However, effectively summarizing longer documents with LLMs still presents some challenges. This blog post dives into the often-overlooked but crucial role of "chunking" and its potential to unlock the full power of LLMs in document summarization, particularly within the context of the Retrieval-Augmented Generation (RAG) model, **powered by the innovative capabilities of MongoDB Atlas Vector Search.**

**RAG and the Chunking Puzzle:**

RAG takes a two-pronged approach to summarization, combining the strengths of both information retrieval and text generation. It first identifies relevant passages within the document based on a query and then uses an LLM to craft a concise and informative summary of those passages. However, the effectiveness of this process hinges heavily on how the document is divided into smaller units, known as "chunks." Chunks that are too large can overwhelm the LLM, leading to inaccurate or incomplete summaries. Conversely, chunks that are too small may not provide enough context for the LLM to understand the overall message of the document.

**The Quest for the Optimal Chunk:**

Researchers have been actively exploring various chunking strategies to optimize the performance of RAG. Here are some:

* **Fixed-size chunks with overlap:** This method involves dividing the document into chunks of a predetermined size, ensuring sufficient context while minimizing information loss at chunk boundaries. By leveraging the $vectorSearch operator, we can now perform efficient Approximate Nearest Neighbor (ANN) searches within each chunk, ensuring we retrieve the most relevant passages for summarization.
* **Recursive chunking:** This strategy takes an iterative approach, starting with the entire document and then splitting each chunk into smaller and smaller pieces. This allows for fine-grained control over the level of detail and context presented to the LLM. MongoDB Atlas Vector Search's vector representation of document content empowers us to perform hierarchical chunking, efficiently identifying the most relevant sub-topics within each segment.
* **Paragraph-based chunking:** This method utilizes natural paragraph breaks to define chunk boundaries, making it suitable for documents with well-defined paragraphs. However, it may not be ideal for texts with more unstructured content. Here, the $filter capabilities of MongoDB Atlas Vector Search come in handy, allowing us to filter chunks based on specific keywords or semantic similarity to ensure we focus on the most relevant parts of the document.
* **Single-page chunks:** This simple approach uses entire pages as individual chunks. While efficient, it may not capture crucial details or effectively address the limitations of LLM processing capabilities. By leveraging the hybrid search capabilities of MongoDB Atlas Vector Search, we can combine traditional keyword search with vector similarity to achieve optimal chunk retrieval, even for single-page documents.

**Other Strategies**

**Parent Document Retrieval Strategies for RAG:**

The effectiveness of RAG relies heavily on the initial retrieval of relevant passages from the "parent document." Here are some key strategies:

* **Keyword matching:** This traditional approach involves matching keywords from the query to keywords within the document. While simple and efficient, it may not capture the full semantic meaning of the query or the document.
* **Passage embedding and retrieval:** This strategy uses vector representations of both the query and the document passages. This allows for more precise retrieval based on semantic similarity, even if the exact keywords don't match. MongoDB Atlas Vector Search excels at this, enabling efficient and accurate retrieval of relevant passages using the $vectorSearch operator.
* **Hybrid search:** This approach combines keyword matching with passage embedding and retrieval. This leverages the strengths of both methods, ensuring both high recall (finding all relevant passages) and high precision (finding only relevant passages).

```
agg_pipeline = [{
	        "$vectorSearch": {
	            "index":'nested_search_index',
	            "path": "text_embedding",
	            "queryVector": query_vector,
                "limit": k,
	            "numCandidates": k * multiplier,
	            },
	        },
	        },
            {
            "$match": {"sample_question": {"$exists": False}}
            },
	        {
    		"$project": {"text_embedding": 0} 
	        },
	    {
        '$lookup' : {"from": "hnsw_parent_retrieval_example",
                      "localField": "parent_id",
                      "foreignField": "_id",
                      "as": 'parent_documents'
                       }},
        {'$unwind': {"path": "$parent_documents"}},
        {"$limit": k}
]
```

This aggregation pipeline in MongoDB Atlas vector search retrieves relevant documents based on a query vector and performs further filtering and processing. Here's a breakdown of each stage:

**Stage 1: $vectorSearch:**

- **index:** Specifies the name of the vector search index used for retrieval.
- **path:** Defines the path within each document where the text embedding vector is stored (assumed to be "text_embedding").
- **queryVector:** The vector representation of the query used for semantic search.
- **limit:** Maximum number of documents to retrieve (k).
- **numCandidates:** Number of candidate documents to consider before filtering (k * multiplier). This helps ensure enough relevant documents are retrieved even after filtering.

**Stage 2: $match:**

- **"sample_question": {"$exists": False}}:** This filters out documents having a field named "sample_question", ensuring we only deal with documents relevant to the current task.

**Stage 3: $project:**

- **"text_embedding": 0:** Excludes the "text_embedding" field from the output documents, potentially reducing document size and improving efficiency.

**Stage 4: $lookup:**

- **"from": "hnsw_parent_retrieval_example":** Specifies the name of the collection containing parent documents.
- **"localField": "parent_id"**: Identifies the field in the current document that stores the parent document ID.
- **"foreignField": "_id"**: Identifies the field in the parent document collection that stores the document ID.
- **"as": 'parent_documents'**: Defines the alias for the retrieved parent documents in the output.

**Stage 5: $unwind:**

- **{"path": "$parent_documents"}**: "Unwinds" the "parent_documents" array, creating separate documents for each parent document associated with the current document.

**Stage 6: $limit:**

- **"limit": k**: Limits the final output to only the k most relevant documents, potentially based on a combination of vector search relevance and information from the parent documents.

Overall, this pipeline uses vector search to retrieve relevant documents based on a query vector, performs further filtering and exclusion, associates each document with its corresponding parent document, and finally returns the k most relevant documents.

## Beyond Retrieval: Unlocking Deeper Insights with Question-Driven Chunking and LLM Processing

Imagine summarizing a news article about a groundbreaking scientific discovery. You've retrieved a relevant chunk, brimming with technical jargon and intricate concepts. To truly grasp the essence of this discovery and prepare the information for LLM-based summarization, a more proactive approach is needed. Here's where **question-driven chunking**, powered by MongoDB Atlas Vector Search, comes into play.

Instead of passively processing the entire chunk, we can ask a targeted question like: "What are the key implications of this new discovery for the field of medicine?" This simple act transforms the process from passive consumption to active exploration, focusing the LLM's attention on the most relevant information.

**Leveraging the Power of Vector Search:**

Through the magic of MongoDB Atlas Vector Search, both the question and the chunk are embedded into a "semantic landscape." This allows us to search for the sentence within the chunk that best aligns with the question's meaning, regardless of exact word matches. This targeted approach unlocks several key benefits:

* **Enhanced Understanding:** By focusing solely on the relevant answer sentence, the LLM receives the most crucial information, leading to a more accurate and insightful summary.
* **Reduced Workload:** The LLM doesn't have to sift through the entire chunk, minimizing processing time and computational resources.
* **Unveiling Deeper Connections:** Asking questions allows us to uncover hidden insights within the information, generating summaries that go beyond just factual details.

**The Power of LLM Processing:**

Once the answer sentence is extracted through MongoDB Atlas Vector Search, the LLM can be used to further refine and summarize the extracted information. This process involves:

* **Contextualization:** Providing the LLM with additional context, such as the original question, relevant sentences from the chunk, and the desired length and key points for the summary.
* **LLM Processing:** The LLM then leverages its capabilities to extract key information, rephrase the answer sentence for clarity and conciseness, and ultimately generate a concise and informative summary.
* **Integration:** This LLM-generated summary can be integrated into a larger summarization system that combines summaries from multiple chunks, performs fact-checking, and offers different summarization styles for diverse audiences and purposes.

**A New Frontier for Text Analysis:**

By combining the power of question-driven chunking with LLM processing, we unlock a new level of sophistication in text analysis and summarization. This approach allows us to:

* Extract the most relevant and insightful information from complex documents.
* Generate summaries that are not only factually accurate but also tailored to specific needs and goals.
* Open up exciting possibilities for utilizing LLM technology for a wide range of applications.

This is not just about summarizing text; it's about unlocking deeper understanding and transforming information into meaningful insights. By embracing a question-driven approach and leveraging the power of LLM processing, we open a new chapter in the field of text analysis and summarization, paving the way for a more insightful and impactful future.

**Beyond Chunking: LLM-powered Enhancements:**

Several innovative approaches leverage LLMs to further improve chunking effectiveness, all powered by MongoDB Atlas Vector Search:

* **LLM pre-summarization:** This strategy involves using an LLM to pre-summarize the content of each chunk before feeding it to the main RAG model. This significantly reduces the workload for the LLM and can lead to more accurate summarization. By storing pre-summarized content as vectors within MongoDB Atlas Vector Search, we can further enhance query efficiency and enable efficient retrieval of relevant chunks.
* **Static text generation from structured data:** This technique leverages LLMs to generate a static textual representation of the information within each chunk. This can be particularly useful for summarizing documents containing complex data structures, such as tables or figures. MongoDB Atlas Vector Search allows us to store and search these generated texts alongside the original data, enabling a more comprehensive understanding of the document's content.
* **Exchange boundary chunking:** This method is specifically designed for dialogue transcripts and involves splitting the transcript based on speaker changes. This allows the LLM to capture the flow of conversation and generate more accurate summaries. In conjunction with MongoDB Atlas Vector Search, we can perform speaker identification and topic segmentation, further optimizing chunk retrieval for dialogue-based content.

## Examples

**Example 1: Fixed-size chunks with overlap and the $vectorSearch operator:**

Imagine you're summarizing a 10-page research paper using the RAG model. Utilizing MongoDB Atlas Vector Search, you can:

1. **Divide the document into fixed-size chunks**, say 1000 words each, with a 500-word overlap. This ensures sufficient context while minimizing information loss at chunk boundaries.
2. **Within each chunk, leverage the $vectorSearch operator to perform efficient ANN searches**. This allows you to identify the most relevant sentences within each chunk, based on your query or specific keywords.
3. **Feed these retrieved sentences to the LLM for summarization**, ensuring that the final summary focuses on the most crucial aspects of the document.

**Example 2: Recursive chunking and vector representation:**

Consider summarizing a legal document with complex nested structures. Using MongoDB Atlas Vector Search, you can:

1. **Start by dividing the document into its main sections**.
2. **For each section, utilize the vector representation of its content to identify sub-topics**.
3. **Recursively apply this process**, further dividing each sub-topic into smaller and more focused segments.
4. **This hierarchical chunking approach, powered by vector similarity, ensures that the LLM receives relevant and contextually rich information for summarization.**

**Example 3: Paragraph-based chunking with $filter:**

You want to summarize an online news article. Utilizing MongoDB Atlas Vector Search, you can:

1. **Divide the document into natural paragraph breaks**.
2. **Apply the $filter operator to filter chunks based on specific keywords** related to your query or area of interest.
3. **This ensures that the LLM focuses solely on the most relevant sections of the article**, generating a concise and informative summary.

**Example 4: Single-page chunks and hybrid search:**

You need to summarize a product manual with minimal text but lots of diagrams and figures. Using MongoDB Atlas Vector Search, you can:

1. **Treat each page as a single chunk**.
2. **Employ hybrid search capabilities**, combining traditional keyword search with vector similarity.
3. **This allows you to retrieve relevant chunks based on both textual content and visual information embedded within the diagrams and figures**.
4. **The LLM can then process these retrieved chunks to generate a comprehensive and accurate summary of the entire product manual.**

**Example 5: LLM pre-summarization and vector storage:**

Imagine you have a large corpus of scientific articles that you need to summarize regularly. With MongoDB Atlas Vector Search, you can:

1. **Pre-summarize each article using an LLM**.
2. **Store these pre-summarized texts as vectors within MongoDB Atlas Vector Search**.
3. **This allows for efficient query processing and retrieval of relevant summaries**, significantly reducing the workload on the main LLM.
4. **When a new query arrives, you can first search for pre-existing summaries based on vector similarity**. This can potentially provide instant results for common queries, saving valuable time and computational resources.

These are just a few examples of how MongoDB Atlas Vector Search can be used to enhance the effectiveness of chunking in LLM-based summarization tasks. By leveraging its powerful search and storage capabilities, researchers and developers can unlock the full potential of LLMs and achieve even better performance in document understanding and summarization.


**The Future of Chunking: Unlocking the Full Potential of LLMs:**

Chunking experimentation, empowered by MongoDB Atlas Vector Search, is an exciting field with the potential to revolutionize the way LLMs approach document summarization. By exploring and optimizing different chunking strategies, researchers are paving the way for LLMs to generate informative, accurate, and concise summaries of even the most complex documents.
