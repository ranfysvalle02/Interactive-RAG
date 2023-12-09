# Understanding the Limitations of Large Language Models (LLMs) and Introducing Retrieval-Augmented Generation (RAG)  
   
As the field of natural language processing continues to evolve, large language models (LLMs) have emerged as a significant breakthrough. Trained on vast amounts of text data, LLMs possess the ability to generate human-like text, revolutionizing various applications. However, it is crucial to comprehend the limitations associated with these models. In this blog post, we will delve into the challenges presented by LLMs and explore interactive retrieval-augmented generation (RAG) architecture as a solution.  
   
## Limitations of LLMs  
   
While LLMs have proven to be incredibly powerful, they do have some drawbacks that need to be acknowledged:  
   
### 1. Hallucinations  
   
One of the limitations of LLMs is their propensity to generate factually inaccurate or ungrounded information, often referred to as "hallucinations." This can pose a challenge in real-world applications where precision and accuracy are paramount.  
   
### 2. Stale Data  
   
LLMs are trained on static datasets that were only current up to a specific point in time. Consequently, they may lack awareness of recent events or developments that occurred after the training data was collected. This temporal limitation can hinder their ability to provide up-to-date information.  
   
### 3. Limited Access to User's Data  
   
LLMs do not have access to a user's local data or personal databases. They rely solely on the knowledge acquired during training, which restricts their capacity to provide personalized or context-specific responses. This limitation can hamper the user experience, particularly when dealing with highly specific or individualized queries.  
   
### 4. Token Limits  
   
LLMs have a maximum token limit, which determines the amount of text they can process in a single interaction. Tokens can represent individual characters, words, subwords, or larger linguistic units. This constraint, such as the token limit of 4096 in OpenAI's gpt-3.5-turbo, can pose challenges when dealing with lengthy or complex queries.  
   
## Introducing Retrieval-Augmented Generation (RAG)  
   
To address these limitations, the retrieval-augmented generation (RAG) architecture was developed. RAG combines the power of vector search, embeddings, and generative AI to enhance the capabilities of LLMs. Here's how RAG overcomes the challenges posed by LLMs:  
   
### 1. Minimizing Hallucinations  
   
By incorporating vector search and retrieval techniques, RAG grounds the generated text in factual information from relevant documents. This approach significantly reduces the occurrence of hallucinations and improves the overall accuracy of the LLM's responses.  
   
### 2. Keeping Information Up-to-Date  
   
RAG leverages vector search to retrieve information from up-to-date sources. By incorporating recent documents, RAG ensures that the LLM's responses reflect the most current and accurate information available, mitigating the issue of stale data.  
   
### 3. Enhanced Personalization  
   
Although LLMs do not have direct access to a user's local data, RAG allows them to utilize external databases or knowledge bases. This enables the inclusion of user-specific information and facilitates more personalized responses, overcoming the limitation of limited access to user data.  
   
### 4. Optimized Token Usage  
   
While RAG does not increase the token limit of an LLM, it optimizes token usage by retrieving only the most relevant documents for generating a response. This ensures that the limited token capacity of LLMs is utilized efficiently, enabling more effective and comprehensive answers.  
   
## Leveraging RAG with Atlas Vector Search  
   
To demonstrate the practical application of the RAG architecture, we will explore how it can be leveraged with Atlas Vector Search, and ActionWeaver to build a AI Agent for interactive retrieval augmented generation. 

By combining RAG with Atlas Vector Search and ActionWeaver, we can build a simple question-answering application that operates on your own terms. 

## RAG Strategy
## ![Alt text](./images/rag.png)
When using GenAI, users may encounter limitations when asking questions that require information not covered in the LLM's training. This can result in incorrect or evasive answers. RAG helps fill these knowledge gaps by treating the question-answering task like an "open-book quiz."

This is the "magic" that empowers the LLM to act as an agent on your behalf, and change the configuration.

```
    @action("iRAG", stop=True)
    def iRAG(self, num_sources:int, chunk_size: int, unique_sources: bool, min_rel_threshold: float):
        """
        Invoke this ONLY when the user asks you to change the RAG configuration.

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
```