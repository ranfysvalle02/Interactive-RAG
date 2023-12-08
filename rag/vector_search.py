import pymongo
import utils

def recall(
        self, text, n_docs=2, min_rel_score=0.25, chunk_max_length=800, unique=True
    ):
        # $vectorSearch
        utils.print_log("recall (VectorSearch)=>" + str(text))
     
        try: 
            response = self.collection.aggregate(
                [
                    {
                        "$vectorSearch": {
                            "index": "default",
                            "queryVector": self.gpt4all_embd.embed_query(text),
                            "path": "embedding",
                            # "filter": {},
                            "limit": 15,  # Number (of type int only) of documents to return in the results. Value can't exceed the value of numCandidates.
                            "numCandidates": 50,  # Number of nearest neighbors to use during the search. You can't specify a number less than the number of documents to return (limit).
                        }
                    },
                    {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                    {"$match": {"score": {"$gte": min_rel_score}}},
                    {"$project": {"score": 1, "_id": 0, "source": 1, "text": 1}},
                ]
            )

        except pymongo.errors.OperationFailure as ex:  
            err_type = type(ex).__name__  
            err_args = ex.args  
            message = f"<b>Error! Please verify Atlas Search index exists.</b><hr/> An exception of type {err_type} occurred with the following arguments:\n{err_args}"  
            self.st.write(f"<div>{message}</div>", unsafe_allow_html=True)  
            raise  
        except Exception as ex:  
            err_type = type(ex).__name__  
            err_args = ex.args  
            message = f"<b>Error! An exception of type {err_type} occurred with the following arguments:\n{err_args}"  
            self.st.write("<div>{message}</div>", unsafe_allow_html=True)  
            raise  

        tmp_docs = []
        str_response = []

        # Interate over the results
        for d in response:
            utils.print_log("recall (Vector Search) returned " + str(len(tmp_docs)) + " documents")
            if len(tmp_docs) == n_docs:
                break
            if unique and d["source"] in tmp_docs:
                continue
            tmp_docs.append(d["source"])
            str_response.append(
                {
                    "URL": d["source"],
                    "content": d["text"][:chunk_max_length],
                    "score": d["score"],
                }
            )
        kb_output = (
            f"RAG Knowledgebase Results[{len(tmp_docs)}]:\n```{str(str_response)}```\n## \n```SOURCES: "
            + str(tmp_docs)
            + "```\n\n"
        )
        self.st.write(kb_output)
        return str(kb_output)