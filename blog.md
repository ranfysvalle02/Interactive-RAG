# Interactive RAG with MongoDB Atlas + Function Calling API
## Introduction: Unveiling the Power of Interactive Knowledge Discovery

Imagine yourself as a detective investigating a complex case. Traditional retrieval-augmented generation (RAG) acts as your static assistant, meticulously sifting through mountains of evidence based on a pre-defined strategy. While helpful, this approach lacks the flexibility needed for today's ever-changing digital landscape.

Enter interactive RAG – the next generation of information access. It empowers users to become active knowledge investigators by:

* **Dynamically adjusting retrieval strategies:** Tailor the search to your specific needs by fine-tuning parameters like the number of sources, chunk size, and retrieval algorithms.
* **Staying ahead of the curve:**  As new information emerges, readily incorporate it into your retrieval strategy to stay up-to-date and relevant.
* **Enhancing LLM performance:** Optimize the LLM's workload by dynamically adjusting the information flow, leading to faster and more accurate analysis.

Before you continue, make sure you understand the basics of:

- [LLMs](https://www.mongodb.com/basics/large-language-models).
- [ RAG](https://www.mongodb.com/basics/retrieval-augmented-generation).
- [Using a vector database](https://www.mongodb.com/basics/vector-databases).

## ![Retrieval Augmented Generation - Diagram 1](./images/RAG-chunks.png)
(_image from  [Session 7: RAG Evaluation with RAGAS and How to Improve Retrieval](https://www.youtube.com/watch?v=mEv-2Xnb_Wk))_

## Optimizing your retrieval strategy: static vs. interactive RAG

Choosing between static and interactive retrieval-augmented generation approaches is crucial for optimizing your application's retrieval strategy. Each approach offers unique advantages and disadvantages, tailored to specific use cases:

**Static RAG:** A static RAG approach is pre-trained on a fixed knowledge base, meaning the information it can access and utilize is predetermined and unchanging. This allows for faster inference times and lower computational costs, making it ideal for applications requiring real-time responses, such as chatbots and virtual assistants.

**Pros:**

* **Faster response:** Pre-loaded knowledge bases enable rapid inference, ideal for real-time applications like chatbots and virtual assistants.
* **Lower cost:** Static RAG requires fewer resources for training and maintenance, making it suitable for resource-constrained environments.
* **Controlled content:** Developers have complete control over the model's knowledge base, ensuring targeted and curated responses in sensitive applications.
* **Consistent results:** Static RAG provides stable outputs even when underlying data changes, ensuring reliability in data-intensive scenarios.

**Cons:**

* **Limited knowledge:** Static RAG is confined to its pre-loaded knowledge, limiting its versatility compared to interactive RAG accessing external data.
* **Outdated information:** Static knowledge bases can become outdated, leading to inaccurate or irrelevant responses if not frequently updated.
* **Less adaptable:** Static RAG can struggle to adapt to changing user needs and preferences, limiting its ability to provide personalized or context-aware responses.

**Interactive RAG:** An interactive RAG approach is trained on a dynamic knowledge base, allowing it to access and process real-time information from external sources such as online databases and APIs. This enables it to provide up-to-date and relevant responses, making it suitable for applications requiring access to constantly changing data.

**Pros:**

* **Up-to-date information:** Interactive RAG can access and process real-time external information, ensuring current and relevant responses, which is particularly valuable for applications requiring access to frequently changing data.
* **Greater flexibility:** Interactive RAG can adapt to user needs and preferences by incorporating feedback and interactions into their responses, enabling personalized and context-aware experiences.
* **Vast knowledge base:** Access to external information provides an almost limitless knowledge pool, allowing interactive RAG to address a wider range of queries and deliver comprehensive and informative responses.

**Cons:**

* **Slower response:** Processing external information increases inference time, potentially hindering real-time applications.
* **Higher cost:** Interactive RAG requires more computational resources, making it potentially unsuitable for resource-constrained environments.
* **Bias risk:** External information sources may contain biases or inaccuracies, leading to biased or misleading responses if not carefully mitigated.
* **Security concerns:** Accessing external sources introduces potential data security risks, requiring robust security measures to protect sensitive information.

### Choosing the right approach

While this tutorial focuses specifically on interactive RAG, the optimal approach depends on your application's specific needs and constraints. Consider:

* **Data size and update frequency:** Static models are suitable for static or infrequently changing data, while interactive RAG is necessary for frequently changing data.
* **Real-time requirements:** Choose static RAG for applications requiring fast response times. For less critical applications, interactive RAG may be preferred.
* **Computational resources:** Evaluate your available resources when choosing between static and interactive approaches.
* **Data privacy and security:** Ensure your chosen approach adheres to all relevant data privacy and security regulations.


## Chunking: a hidden hero in the rise of GenAI

Now, let's put our detective hat back on. If you have a mountain of evidence available for a particular case, you wouldn't try to analyze every piece of evidence at once, right? You'd break it down into smaller, more manageable pieces — documents, witness statements, physical objects — and examine each one carefully. In the world of large language models, this process of breaking down information is called _chunking_, and it plays a crucial role in unlocking the full potential of retrieval-augmented generation.

Just like a detective, an LLM can't process a mountain of information all at once. Chunking helps it break down text into smaller, more digestible pieces called _chunks_. Think of these chunks as bite-sized pieces of knowledge that the LLM can easily analyze and understand. This allows the LLM to focus on specific sections of the text, extract relevant information, and generate more accurate and insightful responses.

However, the size of each chunk isn't just about convenience for the LLM; it also significantly impacts the _retrieval vector relevance score_, a key metric in evaluating the effectiveness of chunking strategies. The process involves converting text to vectors, measuring the distance between them, utilizing ANN/KNN algorithms, and calculating a score for the generated vectors.

Here is an example: Imagine asking "What is a mango?" and the LLM dives into its knowledge base, encountering these chunks:

**High scores:**

* **Chunk:** "Mango is a tropical stone fruit with a sweet, juicy flesh and a single pit." (Score: 0.98)
* **Chunk:** "In India, mangoes are revered as the 'King of Fruits' and hold cultural significance." (Score: 0.92)
* **Chunk:** "The mango season brings joy and delicious treats like mango lassi and mango ice cream." (Score: 0.85)

These chunks directly address the question, providing relevant information about the fruit's characteristics, cultural importance, and culinary uses. High scores reflect their direct contribution to answering your query.

**Low scores:**

* **Chunk:** "Volcanoes spew molten lava and ash, causing destruction and reshaping landscapes." (Score: 0.21)
* **Chunk:** "The stock market fluctuates wildly, driven by economic factors and investor sentiment." (Score: 0.42)
* **Chunk:** "Mitochondria, the 'powerhouses of the cell,' generate energy for cellular processes." (Score: 0.55)

These chunks, despite containing interesting information, are completely unrelated to mangoes. They address entirely different topics, earning low scores due to their lack of relevance to the query.

Check out [ChunkViz v0.1](https://www.chunkviz.com/) to get a feel for how chunk size (character length) breaks down text.

![Chunk Visualization](./images/chunkviz-1.png)

**Balancing detail and context:**

The size of each chunk influences the retrieval vector relevance score in distinct ways:

**Smaller chunk size:**

* **Pros:**
    * Precise focus on specific details and nuances
    * Potentially higher relevance scores due to accurate information extraction
    * Increased sensitivity to subtle changes in meaning
* **Cons:**
    * May sacrifice broader context and understanding of the overall message
    * Requires more computational resources to process numerous chunks
    * Increased risk of missing relevant information due to limited context

**Larger chunk size:**

* **Pros:**
    * Provides a richer context for comprehending the overall message
    * More efficient processing with fewer chunks to handle
    * Potentially higher relevance scores for related chunks due to broader context
* **Cons:**
    * May overlook specific details and subtle shifts in meaning
    * Increased risk of including irrelevant information within a chunk, potentially lowering the relevance score

**Examples in action:**

**Smaller chunk size:**

* **Example:** Analyzing specific clauses in a legal document to identify potential inconsistencies
* **Benefit:** Increased precision in detecting subtle shifts in meaning and ensuring accurate retrieval of relevant information

**Larger chunk size:**

* **Example:** Summarizing a long document by extracting key ideas and information across various sections
* **Benefit:** Improved context for comprehending the overall message and the relationships between different parts of the text

**Considerations for optimal chunking:**

Finding the ideal chunk size is a delicate balance between focusing on specific details and capturing the broader context. Several factors influence this:

* **Task at hand:** For tasks like question-answering, smaller chunks might be preferred for pinpoint accuracy. In contrast, summarization tasks benefit from larger chunks for better context.
* **Data type:** Different types of data might require different chunking approaches. For example, code might be chunked differently than a news article.
* **Desired accuracy:** Smaller chunks can lead to higher precision, while larger chunks might offer better overall understanding.

**Unlocking the future:**

Effective chunking maximizes the retrieval vector relevance score, enabling LLMs to generate the most accurate and insightful responses possible. By understanding the impact of chunk size and other relevant factors, we can unleash the full potential of LLMs and unlock exciting opportunities for the future. In this tutorial, the chunk size we will be controlling interactively is the retrieval chunk.

## Interactive retrieval-augmented generation

## ![RAG Agent Architecture for this Tutorial](./images/rag-agent.png)

In this tutorial, we will showcase an interactive RAG agent. An agent is a computer program or system designed to perceive its environment, make decisions, and achieve specific goals. The interactive RAG agent we will showcase supports the following actions:
- answering questions
- searching the web
- reading web content (URLs)
- listing all sources
- removing sources
- resetting messages
- modifying rag strategy (num_sources, chunk_size, etc.)

## Taking control with interactive RAG

While an optimized chunk size is crucial, interactive RAG goes a step further. It empowers users to dynamically adjust their RAG strategy in real-time, using the function calling API of large language models. This unlocks a new era of personalized information access and knowledge management.

This interactive RAG tutorial leverages:

* **Dynamic strategy adjustment:** Unlike traditional RAG approaches, users can fine-tune chunk size, the number of sources, and other parameters on the fly, tailoring the LLM's response to their specific needs.
* **Function calling API integration:** Function calling API seamlessly integrates external tools and services with LLMs. This allows users to seamlessly incorporate their data sources and tools into their RAG workflow.

**Benefits:**

* Enhanced information retrieval and knowledge management
* Improved accuracy and relevance of LLM responses
* Flexible and versatile framework for building AI applications


## Ingesting content into your vector database

### Streamlining content ingestion with function calling

While vector databases offer significant advantages for GenAI applications, the process of ingesting content can feel cumbersome. Fortunately, we can harness the power of function calling API to seamlessly add new content to the database, simplifying the workflow and ensuring continuous updates.

### Choosing the right home for your embeddings

While various databases can store vector embeddings, each with unique strengths, [MongoDB Atlas](https://cloud.mongodb.com) stands out for GenAI applications. Imagine MongoDB as a delicious cake you can both bake and eat. Not only does it offer the familiar features of MongoDB, but it also lets you store and perform mathematical operations on your vector embeddings directly within the platform. This eliminates the need for separate tools and streamlines the entire process.

By leveraging the combined power of function calling API and MongoDB Atlas, you can streamline your content ingestion process and unlock the full potential of vector embeddings for your GenAI applications.

![RAG architecture diagram with MongoDB Atlas](./images/mdb_diagram.png)

### Detailed breakdown  
   
1. **Vector embeddings**: MongoDB Atlas provides the functionality to store vector embeddings at the core of your document. These embeddings are generated by converting text, video, or audio into vectors utilizing models such as [GPT4All](https://gpt4all.io/index.html), [OpenAI](https://openai.com/) or [Hugging Face](https://huggingface.co/).

    ```python
   	 # Chunk Ingest Strategy
   	 self.text_splitter = RecursiveCharacterTextSplitter(
            	# Set a really small chunk size, just to show.
            	chunk_size=4000, # THIS CHUNK SIZE IS FIXED - INGEST CHUNK SIZE DOES NOT CHANGE
            	chunk_overlap=200, # CHUNK OVERLAP IS FIXED
            	length_function=len,
            	add_start_index=True,
   	 )
   	 # load data from webpages using Playwright. One document will be created for each webpage
   	 # split the documents using a text splitter to create "chunks"
   	 loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])  
   	 documents = loader.load_and_split(self.text_splitter)
   	 self.index.add_documents(
       		 documents
   	 )   
    ```

2. **Vector index**: When employing vector search, it's necessary to [create a search index](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/). This process entails setting up the vector path, aligning the dimensions with your chosen model, and selecting a vector function for searching the top K-nearest neighbors.  
    ```python
    	{
        	"name": "<index-name>",
        	"type": "vectorSearch",
        	"fields":[
            	{
            	"type": "vector",
            	"path": <field-to-index>,
            	"numDimensions": <number-of-dimensions>,
            	"similarity": "euclidean | cosine | dotProduct"
            	},
            	...
        	]
    	}
    ```
3. **Chunk retrieval**: Once the vector embeddings are indexed, an aggregation pipeline can be created on your embedded vector data to execute queries and retrieve results. This is accomplished using the [$vectorSearch](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage) operator, a new aggregation stage in Atlas.

    ```python
    def recall(self, text, n_docs=2, min_rel_score=0.25, chunk_max_length=800,unique=True):
   		 #$vectorSearch
   		 print("recall=>"+str(text))
   		 response = self.collection.aggregate([
   		 {
       		 "$vectorSearch": {
           		 "index": "default",
           		 "queryVector": self.gpt4all_embd.embed_query(text), #GPT4AllEmbeddings()
           		 "path": "embedding",
           		 #"filter": {},
           		 "limit": 15, #Number (of type int only) of documents to return in the results. Value can't exceed the value of numCandidates.
           		 "numCandidates": 50 #Number of nearest neighbors to use during the search. You can't specify a number less than the number of documents to return (limit).
       		 }
   		 },
   		 {
       		 "$addFields":
       		 {
           		 "score": {
           		 "$meta": "vectorSearchScore"
       		 }
   		 }
   		 },
   		 {
       		 "$match": {
           		 "score": {
           		 "$gte": min_rel_score
       		 }
   		 }
   		 },{"$project":{"score":1,"_id":0, "source":1, "text":1}}])
   		 tmp_docs = []
   		 str_response = []
   		 for d in response:
       		 if len(tmp_docs) == n_docs:
           		 break
       		 if unique and d["source"] in tmp_docs:
           		 continue
       		 tmp_docs.append(d["source"])
       		 str_response.append({"URL":d["source"],"content":d["text"][:chunk_max_length],"score":d["score"]})
   		 kb_output = f"Knowledgebase Results[{len(tmp_docs)}]:\n```{str(str_response)}```\n## \n```SOURCES: "+str(tmp_docs)+"```\n\n"
   		 self.st.write(kb_output)
   		 return str(kb_output)
    ```

In this tutorial, we will mainly be focusing on the **CHUNK RETRIEVAL** strategy using the function calling API of LLMs and MongoDB Atlas as our **[data platform](https://www.mongodb.com/atlas)**.

## Key features of MongoDB Atlas
MongoDB Atlas offers a robust vector search platform with several key features, including:

1. **$vectorSearch operator:**
This powerful aggregation pipeline operator allows you to search for documents based on their vector embeddings. You can specify the index to search, the query vector, and the similarity metric to use. [$vectorSearch](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage) provides efficient and scalable search capabilities for vector data.

2. **Flexible filtering:**
You can combine $vectorSearch with other aggregation pipeline operators like [$match](https://www.mongodb.com/docs/v7.0/reference/operator/aggregation/match/), [$sort](https://www.mongodb.com/docs/v7.0/reference/operator/aggregation/sort/), and [$limit](https://www.mongodb.com/docs/v7.0/reference/operator/aggregation/limit/) to filter and refine your search results. This allows you to find the most relevant documents based on both their vector embeddings and other criteria.

3. **Support for various similarity metrics:**
MongoDB Atlas supports different similarity metrics like [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) and [euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance), allowing you to choose the best measure for your specific data and task.

4. **High performance:**
The vector search engine in MongoDB Atlas is optimized for large datasets and high query volumes, ensuring efficient and responsive search experiences.

5. **Scalability:**
MongoDB Atlas scales seamlessly to meet your growing needs, allowing you to handle increasing data volumes and query workloads effectively.

**Additionally, MongoDB Atlas offers several features relevant to its platform capabilities:**

* **Global availability:**
Your data is stored in multiple data centers around the world, ensuring high availability and disaster recovery.
* **Security:**
MongoDB Atlas provides robust security features, including encryption at rest and in transit, access control, and data audit logging.
* **Monitoring and alerting:**
MongoDB Atlas provides comprehensive monitoring and alerting features to help you track your cluster's performance and identify potential issues.
* **Developer tools:**
MongoDB Atlas offers various developer tools and APIs to simplify development and integration with your applications.

## OpenAI function calling:
OpenAI's function calling is a powerful capability that enables users to seamlessly interact with OpenAI models, such as GPT-3.5, through programmable commands. This functionality allows developers and enthusiasts to harness the language model's vast knowledge and natural language understanding by incorporating it directly into their applications or scripts. Through function calling, users can make specific requests to the model, providing input parameters and receiving tailored responses. This not only facilitates more precise and targeted interactions but also opens up a world of possibilities for creating dynamic, context-aware applications that leverage the extensive linguistic capabilities of OpenAI's models. Whether for content generation, language translation, or problem-solving, OpenAI function calling offers a flexible and efficient way to integrate cutting-edge language processing into various domains.

## Key features of OpenAI function calling:
- Function calling allows you to connect large language models to external tools.
- The [Chat Completions API](https://platform.openai.com/docs/guides/text-generation/chat-completions-api) generates JSON that can be used to call functions in your code.
- The latest models have been trained to detect when a function should be called and respond with JSON that adheres to the function signature.
- Building user confirmation flows is recommended before taking actions that impact the world on behalf of users.
- Function calling can be used to create assistants that answer questions by calling external APIs, convert natural language into API calls, and extract structured data from text.
- The basic sequence of steps for function calling involves calling the model, parsing the JSON response, calling the function with the provided arguments, and summarizing the results back to the user.
- Function calling is supported by specific model versions, including [GPT-4](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo) and [GPT-3.5-turbo](https://platform.openai.com/docs/models/gpt-3-5).
- Parallel function calling allows multiple function calls to be performed together, reducing round-trips with the API.
- Tokens are used to inject functions into the system message and count against the model's context limit and billing.

![Function Calling Diagram - Simple](./images/function_calling.png)

Read more at ThinhDA.

## Function calling API basics: actions

Actions are functions that an agent can invoke. There are two important design considerations around actions:

    * Giving the agent access to the right actions
    * Describing the actions in a way that is most helpful to the agent

## Crafting actions for effective agents

**Actions are the lifeblood of an agent's decision-making.** They define the options available to the agent and shape its interactions with the environment. Consequently, designing effective actions is crucial for building successful agents.

Two key considerations guide this design process:

1. **Access to relevant actions:** Ensure the agent has access to actions necessary to achieve its objectives. Omitting critical actions limits the agent's capabilities and hinders its performance.
2. **Action description clarity:** Describe actions in a way that is informative and unambiguous for the agent. Vague or incomplete descriptions can lead to misinterpretations and suboptimal decisions.

By carefully designing actions that are both accessible and well-defined, you equip your agent with the tools and knowledge necessary to navigate its environment and achieve its objectives.

Further considerations:

* **Granularity of actions:** Should actions be high-level or low-level? High-level actions offer greater flexibility but require more decision-making, while low-level actions offer more control but limit adaptability.
* **Action preconditions and effects:** Clearly define the conditions under which an action can be taken and its potential consequences. This helps the agent understand the implications of its choices.


If you don't give the agent the right actions and describe them in an effective way, you won’t be able to build a working agent.

![LangChain Tools Diagram](./images/llm_agent.png)
(_Credit to blog post: [Make Langchain Agent Actually Work With Local LLMs (Vicuna, WizardLM)](https://betterprogramming.pub/make-langchain-agent-actually-works-with-local-llms-vicuna-wizardlm-etc-da42b6b1a97)_)

An LLM is then called, resulting in either a response to the user or action(s) to be taken. If it is determined that a response is required, then that is passed to the user, and that cycle is finished. If it is determined that an action is required, that action is then taken, and an observation (action result) is made. That action and corresponding observation are added back to the prompt (we call this an “agent scratchpad”), and the loop resets — i.e., the LLM is called again (with the updated agent scratchpad).

## Getting started

Clone the demo Github repository.
```bash
git clone git@github.com:ranfysvalle02/Interactive-RAG.git
```

Create a new Python environment.
```bash
python3 -m venv env
```

Activate the new Python environment.
```bash
source env/bin/activate
```

Install the requirements.
```bash
pip3 install -r requirements.txt
```
Set the parameters in [params.py](rag/params.py):
```bash
# MongoDB
MONGODB_URI = ""
DATABASE_NAME = "genai"
COLLECTION_NAME = "rag"

# If using OpenAI
OPENAI_API_KEY = ""

# If using Azure OpenAI
#OPENAI_TYPE = "azure"
#OPENAI_API_VERSION = "2023-10-01-preview"
#OPENAI_AZURE_ENDPOINT = "https://.openai.azure.com/"
#OPENAI_AZURE_DEPLOYMENT = ""

```
Create a Search index with the following definition:
```JSON
{
  "type": "vectorSearch",
  "fields": [
	{
  	"numDimensions": 384,
  	"path": "embedding",
  	"similarity": "cosine",
  	"type": "vector"
	}
  ]
}
```

Set the environment.
```bash
export OPENAI_API_KEY=
```

To run the RAG application:

```bash
env/bin/streamlit run rag/app.py
```
Log information generated by the application will be appended to app.log.

## Usage
This bot supports the following actions: answering questions, searching the web, reading URLs, removing sources, listing all sources, viewing messages, and resetting messages.

It also supports an action called iRAG that lets you dynamically control your agent's RAG strategy.

Ex: "set RAG config to 3 sources and chunk size 1250" => New RAG config:{'num_sources': 3, 'source_chunk_size': 1250, 'min_rel_score': 0, 'unique': True}.

If the bot is unable to provide an answer to the question from data stored in the Atlas Vector store and your RAG strategy (number of sources, chunk size, min_rel_score, etc), it will initiate a web search to find relevant information. You can then instruct the bot to read and learn from those results.


## Demo

Let's start by asking our agent a question — in this case, "What is a mango?" The first thing that will happen is it will try to "recall" any relevant information using vector embedding similarity. It will then formulate a response with the content it "recalled" or will perform a web search. Since our knowledge base is currently empty, we need to add some sources before it can formulate a response.

![DEMO - Ask a Question](./images/ask_question.png)

Since the bot is unable to provide an answer using the content in the vector database, it initiated a Google search to find relevant information. We can now tell it which sources it should "learn." In this case, we'll tell it to learn the first two sources from the search results.



![DEMO - Add a source](./images/add_sources.png)

## Change RAG strategy

Next, let's modify the RAG strategy! Let's make it only use one source and have it use a small chunk size of 500 characters.

![DEMO - Change RAG strategy part 1](./images/mod_rag.png)

Notice that though it was able to retrieve a chunk with a fairly high relevance score, it was not able to generate a response because the chunk size was too small and the chunk content was not relevant enough to formulate a response. Since it could not generate a response with the small chunk, it performed a web search on the user's behalf.

Let's see what happens if we increase the chunk size to 3,000 characters instead of 500.

![DEMO - Change RAG strategy part 2](./images/mod_rag-2.png)

Now, with a larger chunk size, it was able to accurately formulate the response using the knowledge from the vector database!

## List all sources

Let's see what's available in the knowledge base of the agent by asking it, “What sources do you have in your knowledge base?”

![DEMO - List all sources](./images/list_sources.png)

## Remove a source of information

If you want to remove a specific resource, you could do something like:
```
USER: remove source 'https://www.oracle.com' from the knowledge base
```

To remove all the sources in the collection, we could do something like:

![DEMO - Remove ALL sources](./images/forget.png)

This demo has provided a glimpse into the inner workings of our AI agent, showcasing its ability to learn and respond to user queries in an interactive manner. We've witnessed how it seamlessly combines its internal knowledge base with real-time web search to deliver comprehensive and accurate information. The potential of this technology is vast, extending far beyond simple question-answering. None of this would be possible without the magic of the function calling API.

## Embracing the future of information access with interactive RAG

This post has explored the exciting potential of interactive retrievalaugmented generation (RAG) with the powerful combination of MongoDB Atlas and function calling API. We've delved into the crucial role of chunking, embedding, and retrieval vector relevance score in optimizing RAG performance, unlocking its true potential for information retrieval and knowledge management.

Interactive RAG, powered by the combined forces of MongoDB Atlas and function calling API, represents a significant leap forward in the realm of information retrieval and knowledge management. By enabling dynamic adjustment of the RAG strategy and seamless integration with external tools, it empowers users to harness the full potential of LLMs for a truly interactive and personalized experience.

Intrigued by the possibilities? Explore the full source code for the interactive RAG application and unleash the power of RAG with MongoDB Atlas and function calling API in your own projects!

Together, let's unlock the transformative potential of this potent combination and forge a future where information is effortlessly accessible and knowledge is readily available to all.

View is the [full source code](https://github.com/ranfysvalle02/Interactive-RAG/) for the interactive RAG application using MongoDB Atlas and function calling API.

### Additional MongoDB Resources

- [RAG with Atlas Vector Search, LangChain, and OpenAI](https://www.mongodb.com/developer/products/atlas/rag-atlas-vector-search-langchain-openai/)
- [Taking RAG to Production with the MongoDB Documentation AI Chatbot](https://www.mongodb.com/developer/products/atlas/taking-rag-to-production-documentation-ai-chatbot/)
- [What is Artificial Intelligence (AI)?](https://www.mongodb.com/basics/what-is-artificial-intelligence)
- [Unlock the Power of Semantic Search with MongoDB Atlas Vector Search](https://www.mongodb.com/basics/semantic-search)
- [Machine Learning in Healthcare:
Real-World Use Cases and What You Need to Get Started](https://www.mongodb.com/basics/machine-learning-healthcare)
- [What is Generative AI?
](https://www.mongodb.com/basics/generative-ai)




