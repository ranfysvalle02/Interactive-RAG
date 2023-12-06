# UPDATE

Took an outdated Interactive RAG demo and changed it to use MongoDB Atlas (along with a few other changes and prompt engineering).

# FLOW
1. Ask Question
2. Check VectorStore -> If nothing, google. If something, respond.
3. Add/Remove sources conversationally

# VIDEO

[DEMO 1](https://apollo-fv-mneqk.mongodbstitch.com/INTERACTIVE-RAG.mp4)

[DEMO 2](https://apollo-fv-mneqk.mongodbstitch.com/IRAG.mp4)

# Interactive Retrieval Augmented Generation

This demo app uses [ActionWeaver](https://github.com/TengHu/ActionWeaver/tree/main) to organize and orchestrate tools/functions and implement an interactive RAG bot.

This demo app is not intended for use in production environments.

## Getting Started
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
For Google Search
SERPAPI_KEY = ""

# MongoDB 
MONGODB_URI = ""
DATABASE_NAME = "genai"
COLLECTION_NAME = "rag"

# If using OpenAI
OPENAI_API_KEY = ""

# If using Azure for OpenAI
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
This bot supports the following actions: answer question, read URLs, remove sources, list all sources, and reset messages. 
It also supports an action called iRAG that lets you dynamically control your agent's RAG strategy. 

Ex: "set RAG config to 3 sources and chunk size 1250" => New RAG config:{'num_sources': 3, 'source_chunk_size': 1250, 'min_rel_score': 0, 'unique': True}.

```
 def __call__(self, text):
        text = self.preprocess_query(text)
        self.messages += [{"role": "user", "content":text}]
        response = self.llm.create(messages=self.messages, actions = [
            self.read_url,self.answer_question,self.remove_source,self.reset_messages,
            self.iRAG, self.get_sources_list,self.search_web
        ], stream=True)
        return response
```

If the bot is unable to provide an answer to the question from data stored in the Atlas Vector store, and your RAG strategy (number of sources and chunk size) it will initiate a Google search to find relevant information. You can then instruct the bot to read and learn from those results. 


## Example

![](./images/question.png)

Since the bot is unable to provide an answer, it initiated a Google search to find relevant information.

## Tell the bot which results to learn from: 

![](./images/add_sources.png)


## Change RAG strategy
![](./images/mod_rag.png)

## List All Sources
![](./images/list_sources.png)

## Remove a source of information
![](./images/remove_sources.png)

## ActionWeaver Basics: What is an Agent anyway?
Although the term “agents” can be used to describe a wide range of applications, OpenAI’s usage of the term is consistent with our understanding: using the LLM alone to define transition options. This can best be thought of as a loop. Given user input, this loop will be entered. 

![]("https://blog.langchain.dev/content/images/size/w1600/2023/11/CleanShot-2023-11-14-at-19.09.59@2x.png)

An LLM is then called, resulting in either a response to the user OR action(s) to be taken. If it is determined that a response is required, then that is passed to the user, and that cycle is finished. If it is determined that an action is required, that action is then taken, and an observation (action result) is made. That action & corresponding observation are added back to the prompt (we call this an “agent scratchpad”), and the loop resets, ie. the LLM is called again (with the updated agent scratchpad).

Large Language Models (LLMs) are great, but they have some limitations (context window, lack of access to real-time data, inability to interact with APIs, etc). LLM agents help overcome those limitations. The definition of an agent in the AI world depends on who you ask - but in a nutshell an agent empowers the LLM to take action. 

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

## ActionWeaver Basics: actions
Developers can attach ANY Python function as a tool with a simple decorator. In the following example, we introduce action get_sources_list, which will be invoked by the OpenAI API.

ActionWeaver utilizes the decorated method's signature and docstring as a description, passing them along to OpenAI's function API.

ActionWeaver provides a light wrapper that takes care of converting the docstring/decorator information into the correct format for the OpenAI API.

```
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
```

## ActionWeaver Basics: stop=True

stop=True when added to an action means that the LLM will immediately return the function's output, but this also restrict the LLM from making multiple function calls. For instance, if asked about the weather in NYC and San Francisco, the model would invoke two separate functions sequentially for each city. However, with `stop=True`, this process is interrupted once the first function returns weather information  for either NYC or San Francisco, depending on which city it queries first.



For a more in-depth understanding of how this bot works under the hood, please refer to the bot.py file. 
Additionally, you can explore the [ActionWeaver](https://github.com/TengHu/ActionWeaver/tree/main) repository for further details.

## RAG Strategy
## ![Alt text](./images/rag.png)
### CLASSIC RAG: noisy chunks, "spray and pray 🙏"
```
[VERIFIED SOURCES]
{chunk1}
{chunk2}
{chunk3}
{chunk4}
[END VERIFIED SOURCES]
```


# ReAct Prompt Technique + Chain of Thought
Generating reasoning traces allow the model to induce, track, and update action plans, and even handle exceptions.
This example uses ReAct combined with chain-of-thought (CoT).

[Chain of Thought](https://www.promptingguide.ai/techniques/cot)

[Reasoning + Action](https://www.promptingguide.ai/techniques/react)

```
[EXAMPLES]
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
- Action: "remove_source"(['https://www.google.com', 'https://www.example.com'])

- User Input: read https://www.google.com, https://www.example.com
- Thought: I have to think step by step. I should not answer directly, let me check my available actions before responding.
- Observation: I have an action available "read_url".
- Action: "read_url"(['https://www.google.com','https://www.example.com'])
[END EXAMPLES]
```


## Credit
This was inspired by https://github.com/TengHu/Interactive-RAG

## Additional MongoDB Resources

- https://www.mongodb.com/developer/products/atlas/taking-rag-to-production-documentation-ai-chatbot/
- https://www.mongodb.com/basics/what-is-artificial-intelligence
- https://www.mongodb.com/basics/vector-databases
- https://www.mongodb.com/basics/semantic-search
- https://www.mongodb.com/basics/machine-learning-healthcare
- https://www.mongodb.com/basics/generative-ai
- https://www.mongodb.com/basics/large-language-models
- https://www.mongodb.com/basics/retrieval-augmented-generation

## Additional Reading
- https://blog.langchain.dev/openais-bet-on-a-cognitive-architecture/

## Contributing
We welcome contributions from the open-source community.

## License
Apache License 2.0
