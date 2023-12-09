## Interactive RAG with MongoDB Atlas

##### BEFORE YOU CONTINUE, MAKE SURE YOU
- Understand the basics of LLMs
- Understand the basic concept of RAG
- Understand the basics of using a vector database

## ![Alt text](https://cdn.stackoverflow.co/images/jo7n4k8s/production/ef172115fca9aa6b3b99eeb1c749acf9f8c183a0-6000x3150.png?w=1200&h=630&auto=format&dpr=2)

Imagine you're a detective trying to solve a complex case. You wouldn't try to analyze every piece of evidence at once, right? You'd break it down into smaller, more manageable pieces - documents, witness statements, physical objects - and examine each one carefully. In the world of large language models (LLMs), this process of breaking down information is called **chunking**, and it plays a crucial role in unlocking the full potential of Retrieval Augmented Generation (RAG).

Just like a detective, an LLM can't process a mountain of information all at once. Chunking helps it break down text into smaller, more digestible pieces called **chunks**. Think of these chunks as bite-sized pieces of knowledge that the LLM can easily analyze and understand. This allows the LLM to focus on specific sections of the text, extract relevant information, and generate more accurate and insightful responses.

However, the size of each chunk isn't just about convenience for the LLM; it also significantly impacts the **retrieval vector relevance score**, a key metric in evaluating the effectiveness of chunking strategies. This score measures how well the generated vector embeddings for each chunk represent the actual information contained within.

**Balancing Detail and Context:**

The size of each chunk influences the retrieval vector relevance score in distinct ways:

**Smaller Chunk Size:**

* **Pros:**
    * Precise focus on specific details and nuances.
    * Potentially higher relevance scores due to accurate information extraction.
    * Increased sensitivity to subtle changes in meaning.
* **Cons:**
    * May sacrifice broader context and understanding of the overall message.
    * Requires more computational resources to process numerous chunks.
    * Increased risk of missing relevant information due to limited context.

**Larger Chunk Size:**

* **Pros:**
    * Provides a richer context for comprehending the overall message.
    * More efficient processing with fewer chunks to handle.
    * Potentially higher relevance scores for related chunks due to broader context.
* **Cons:**
    * May overlook specific details and subtle shifts in meaning.
    * Increased risk of including irrelevant information within a chunk, potentially lowering the relevance score.

**Examples in Action:**

**Smaller Chunk Size:**

* **Example:** Analyzing specific clauses in a legal document to identify potential inconsistencies.
* **Benefit:** Increased precision in detecting subtle shifts in meaning and ensuring accurate retrieval of relevant information.

**Larger Chunk Size:**

* **Example:** Summarizing a long document by extracting key ideas and information across various sections.
* **Benefit:** Improved context for comprehending the overall message and the relationships between different parts of the text.

**Considerations for Optimal Chunking:**

Finding the ideal chunk size is a delicate balance between focusing on specific details and capturing the broader context. Several factors influence this:

* **Task at hand:** For tasks like question answering, smaller chunks might be preferred for pinpoint accuracy. In contrast, summarization tasks benefit from larger chunks for better context.
* **Data type:** Different types of data might require different chunking approaches. For example, code might be chunked differently than a news article.
* **Desired accuracy:** Smaller chunks can lead to higher precision, while larger chunks might offer better overall understanding.

**Unlocking the Future:**

Effective chunking maximizes the retrieval vector relevance score, enabling LLMs to generate the most accurate and insightful responses possible. By understanding the impact of chunk size and other relevant factors, we can unleash the full potential of LLMs and unlock exciting opportunities for the future.

## Interactive Retrieval Augmented Generation

## ![Alt text](./images/actionweaver_mdb.png)

Ultimately, effective chunking is about finding the right balance between detail and context, maximizing the retrieval vector relevance score and enabling the LLM to generate the most accurate and insightful responses possible. By understanding the impact of chunk size and other relevant factors, we can unleash the full potential of LLMs and unlock exciting possibilities!

## Taking Control with Interactive RAG:

While an optimized chunk size is crucial, Interactive RAG goes a step further. It empowers users to dynamically adjust their RAG strategy in real-time, using ActionWeaver. This unlocks a new era of personalized information access and knowledge management.

**This Interactive RAG tutorial leverages:**

* **Dynamic Strategy Adjustment:** Unlike traditional RAG approaches, users can fine-tune chunk size, number of sources, and other parameters on-the-fly, tailoring the LLM's response to their specific needs.
* **ActionWeaver Integration:** ActionWeaver seamlessly integrates external tools and services with LLMs through function calling. This allows users to seamlessly incorporate their own data sources and tools into their RAG workflow.

**Benefits:**

* Enhanced information retrieval and knowledge management
* Improved accuracy and relevance of LLM responses
* Flexible and versatile framework for building AI applications

## The "Ingest Process"

Why have a separate process to "ingest" your content into your vector database? Using the magic of agents, we can easily add new content to the vector database.

There are many types of databases that can store these embeddings, each with its own special uses. But for tasks involving GenAI applications, I recommend MongoDB.

Think of MongoDB as a cake that you can both have and eat. It gives you the power of its language for making queries, Mongo Query Language. It also includes all the great features of MongoDB. On top of that, it lets you store these building blocks (vector embeddings) and do math operations on them, all in one place. This makes MongoDB Atlas a one-stop shop for all your vector embedding needs!

![](https://www.mongodb.com/developer/_next/image/?url=https%3A%2F%2Fimages.contentstack.io%2Fv3%2Fassets%2Fblt39790b633ee0d5a7%2Fbltb482d06c8f1f0674%2F65398a092c3581197ab3b07f%2Fimage3.png&w=1920&q=75)

### Detailed Breakdown:  
   
1. **Vector Embeddings**: MongoDB Atlas provides the functionality to store vector embeddings at the core of your document. These embeddings are generated by converting text, video, or audio into vectors utilizing models such as GPT4All, OpenAI or Hugging Face.  

```
    # Chunk Ingest Strategy
    self.text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=4000,
        chunk_overlap=200,
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

2. **Indexing**: When employing vector search, it's necessary to create a search index. This process entails setting up the vector path, aligning the dimensions with your chosen model, and selecting a vector function for searching the top K-nearest neighbors.  
```
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 384, #dimensions depends on the model
        "similarity": "cosine",
        "type": "knnVector"
      }
    }
  }
}
```
3. **Chunk Retrieval**: Once the vector embeddings are indexed, an aggregation pipeline can be created on your embedded vector data to execute queries and retrieve results. This is accomplished using the $vectorSearch operator, a new aggregation stage in Atlas.

```
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

In this tutorial/example, we will mainly be focusing on the **CHUNK RETRIEVAL** strategy. 

# Building an Interactive-RAG Agent

Using [ActionWeaver](https://github.com/TengHu/ActionWeaver/tree/main), a lightweight wrapper for function calling API, we can build a user proxy agent that efficiently retrieves and ingests relevant information using MongoDB Atlas. 

A proxy agent is a middleman sending client requests to other servers or resources and then bringing responses back. 

This agent presents the data to the user in an interactive and customizable manner, enhancing the overall user experience.

The `UserProxyAgent` has several RAG parameters that can be customized, such as `chunk_size`(e.g. 1000), `num_sources`(e.g. 2), `unique`(e.g. True) and `min_rel_score`(e.g. 0.00).

```
class UserProxyAgent:
    def __init__(self, logger, st):
        # CHUNK RETRIEVAL STRATEGY
        self.rag_config = {
            "num_sources": 2,
            "source_chunk_size": 1000,
            "min_rel_score": 0.00,
            "unique": True,
        }
```

## Getting Started

Clone the demo Github repository
```
git clone git@github.com:ranfysvalle02/Interactive-RAG.git
```

Create a new Python environment
```bash 
python3 -m venv env
```

Activate the new Python enviroment
```bash
source env/bin/activate
```

Install the requirements
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
OPENAI_TYPE = "azure"
OPENAI_API_VERSION = "2023-10-01-preview"
OPENAI_AZURE_ENDPOINT = "https://.openai.azure.com/"
OPENAI_AZURE_DEPLOYMENT = ""

```
Create a Search index with the following definition
```JSON
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 384,
        "similarity": "cosine",
        "type": "knnVector"
      }
    }
  }
}
```

Set the environment
```bash
export OPENAI_API_KEY=
```

To run the RAG application

```bash
env/bin/streamlit run rag/app.py
```
Log information generated by the application will be appended to app.log.

## Usage
This bot supports the following actions: answer question, search the web, read URLs, remove sources, list all sources, view messages and reset messages. 

It also supports an action called iRAG that lets you dynamically control your agent's RAG strategy. 

Ex: "set RAG config to 3 sources and chunk size 1250" => New RAG config:{'num_sources': 3, 'source_chunk_size': 1250, 'min_rel_score': 0, 'unique': True}.

If the bot is unable to provide an answer to the question from data stored in the Atlas Vector store, and your RAG strategy (number of sources, chunk size, min_rel_score, etc) it will initiate a web search to find relevant information. You can then instruct the bot to read and learn from those results. 


## Example

![](./images/ask_question.png)

Since the bot is unable to provide an answer, it initiated a Google search to find relevant information.

## Tell the bot which results to learn from: 

![](./images/add_sources.png)


## Change RAG strategy
![](./images/mod_rag.png)

## List All Sources
![](./images/list_sources.png)

## Remove a source of information
![](./images/remove_sources.png)

## Why Choose ActionWeaver? 
Here are some key benefits that influenced our decision to choose ActionWeaver:
1. Lightweight and Single-Purposed: ActionWeaver is very lightweight and designed with a singular focus on building LLM applications with function calling. This specialization ensures that it excels in its core function without unnecessary complexity.
2. Ease of Use:  ActionWeaver streamlines the process of integrating external tools into agent's toolkit. Using a simple decorator, developers can effortlessly add any Python function, and it also provides the flexibility to include tools from other ecosystems like LangChain or Llama Index.
3. Versatility: Despite its simplicity, ActionWeaver offers a wide range of capabilities, including support for forced function execution, parallel function calling and structured data extraction. Such versatility makes it a Swiss Army knife, equipped to handle a variety of AI-related tasks and adapt seamlessly to changing project demands.
4. Minimal Dependency: ActionWeaver has minimal dependencies, relying only on the openai and pydantic libraries. This reduces the overhead of managing dependencies.
5. Complex Function Orchestration: The framework empowers us to create intricate sequences of function calls, allowing us to build complex hierarchies or chains of functions. This capability enables us to execute sophisticated workflows with ease. 

## ActionWeaver Basics: What is an Agent anyway?

An agent is basically just a computer program or system designed to perceive its environment, make decisions, and achieve specific goals.

Think of an agent as a software entity that displays some degree of autonomy and performs actions in its environment on behalf of its user or owner, but in a relatively independent way. It takes initiatives to perform actions on its own by deliberating its options to achieve its goal(s). The core idea of agents is to use a language model to choose a sequence of actions to take. In contrast to chains, where a sequence of actions is hardcoded in code, agents use a language model as a reasoning engine to determine which actions to take and in which order.

### Actions

Actions are functions that an agent can invoke. There are two important design considerations around actions:

    Giving the agent access to the right actions
    Describing the actions in a way that is most helpful to the agent

Without thinking through both, you won’t be able to build a working agent. If you don’t give the agent access to a correct set of actions, it will never be able to accomplish the objectives you give it. If you don’t describe the actions well, the agent won’t know how to use them properly.

![](./images/llm_agent.png)

An LLM is then called, resulting in either a response to the user OR action(s) to be taken. If it is determined that a response is required, then that is passed to the user, and that cycle is finished. If it is determined that an action is required, that action is then taken, and an observation (action result) is made. That action & corresponding observation are added back to the prompt (we call this an “agent scratchpad”), and the loop resets, ie. the LLM is called again (with the updated agent scratchpad).

In ActionWeaver, we can influence the loop adding `stop=True|False` to an action. 
If `stop=True`, the LLM will immediately return the function's output. This will also restrict the LLM from making multiple function calls.
In this demo we will only be using `stop=True`

ActionWeaver also supports more complex loop control using `orch_expr(SelectOne[actions])` and `orch_expr(RequireNext[actions])`, but I'll leave that for PART II.

![](./images/scale_tools.png)

The ActionWeaver agent framework is an AI application framework that puts function-calling at its core. It is designed to enable seamless merging of traditional computing systems with the powerful reasoning capabilities of Language Model Models. 
ActionWeaver is built around the concept of LLM function calling, while popular frameworks like Langchain and Haystack are built around the concept of pipelines. 

## Key features of ActionWeaver include:
- Ease of Use: ActionWeaver allows developers to add any Python function as a tool with a simple decorator. The decorated method's signature and docstring are used as a description and passed to OpenAI's function API.
- Function Calling as First-Class Citizen: Function-calling is at the core of the framework.
- Extensibility: Integration of any Python code into the agent's toolbox with a single line of code, including tools from other ecosystems like LangChain or Llama Index.
- Function Orchestration: Building complex orchestration of function callings, including intricate hierarchies or chains.
- Debuggability: Structured logging improves the developer experience.

## Key features of OpenAI functions include:
- Function calling allows you to connect large language models to external tools.
- The Chat Completions API generates JSON that can be used to call functions in your code.
- The latest models have been trained to detect when a function should be called and respond with JSON that adheres to the function signature.
- Building user confirmation flows is recommended before taking actions that impact the world on behalf of users.
- Function calling can be used to create assistants that answer questions by calling external APIs, convert natural language into API calls, and extract structured data from text.
- The basic sequence of steps for function calling involves calling the model, parsing the JSON response, calling the function with the provided arguments, and summarizing the results back to the user.
- Function calling is supported by specific model versions, including gpt-4 and gpt-3.5-turbo.
- Parallel function calling allows multiple function calls to be performed together, reducing round-trips with the API.
- Tokens are used to inject functions into the system message and count against the model's context limit and billing.

![](./images/function_calling.jpeg)

Read more at: https://thinhdanggroup.github.io/function-calling-openai/

## Embracing the Future of Information Access with Interactive RAG

So far, we've explored the intricacies of chunking and its impact on the effectiveness of RAG. Now, let's shift our focus to the revolutionary possibilities unlocked by **Interactive RAG**, a game-changer in the landscape of information access and knowledge management.

Interactive RAG empowers users to move beyond static RAG strategies and dynamically adjust their strategy in real-time. This is achieved through the integration of cutting-edge tools like ActionWeaver, which seamlessly integrates external tools and services with LLMs through function calls.

## Conclusion
### Interactive Retrieval Augmented Generation with MongoDB Atlas and ActionWeaver: A Powerful Synergy

This blog post has explored the exciting potential of **Interactive Retrieval Augmented Generation (RAG)** with the powerful combination of MongoDB Atlas and ActionWeaver. We've delved into the crucial role of **chunking, embedding, and retrieval vector relevance score** in optimizing RAG performance, unlocking its true potential for information retrieval and knowledge management.

Furthermore, we introduced **ActionWeaver**, a lightweight framework that simplifies the integration of external tools with language models through **function calling**. This powerful synergy empowers us to build robust and versatile AI applications, expanding the boundaries of what's possible.


Interactive RAG, powered by the combined forces of MongoDB Atlas and ActionWeaver, represents a significant leap forward in the realm of information retrieval and knowledge management. By enabling dynamic adjustment of the RAG strategy and seamless integration with external tools, it empowers users to harness the full potential of LLMs for a truly interactive and personalized experience.

Intrigued by the possibilities? Explore the full source code for the **Interactive-RAG application** and unleash the power of RAG with MongoDB Atlas and ActionWeaver in your own projects!

**Together, let's unlock the transformative potential of this potent combination and forge a future where information is effortlessly accessible and knowledge is readily available to all.**

![Here is the full source code for the Interactive-RAG application using MongoDB Atlas and ActionWeaver!](https://github.com/ranfysvalle02/Interactive-RAG/)