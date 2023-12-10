## Relative Score Fusion for Enhanced Search Results

This blog post explores a method for combining the power of vector search and full-text search through relative score fusion. This approach utilizes the Sphere dataset, a large corpus for knowledge-intensive NLP tasks.

**The Scenario:**

We aim to search for information about companies from the "names" list using both vector search and full-text search. Vector search leverages sentence embeddings for semantic similarity, while full-text search focuses on keyword matching.

**The Approach:**

1. **Data Setup:**
    - Sentence embeddings are generated for each company name using the Facebook DPR question encoder model.
    - MongoDB collections are used to store the data and facilitate queries.
2. **Pipeline Breakdown:**
    - **Vector Search:**
        - The "$vectorSearch" aggregation operator searches for documents with similar vector representations.
        - The retrieved documents are assigned a "vs_score" based on their search score and scaled using pre-defined parameters.
    - **Full Text Search:**
        - The "$search" operator performs full-text search based on the company name.
        - Matching documents are assigned an "fts_score" and scaled similarly.
    - **Relative Score Fusion:**
        - Both sets of results are combined using "$unionWith".
        - The "$group" operator aggregates the maximum scores for each document across both search methods.
        - The final score is calculated by adding the scaled "vs_score" and "fts_score" for each document.
        - The results are then sorted by the final score in descending order, presenting the most relevant documents first.

**CODE:**

gist available here: https://gist.github.com/hweller1/d6dbd5036ae4366108b534a0f1662a20

```
vector_agg_with_lookup = [
        {
            "$vectorSearch": {
                "index": "vector",
                "path": "vector",
                "queryVector": embedding.tolist(),
                "numCandidates": k * overrequest_factor,
                "limit": k * 2
            }
        },
        {"$addFields": {"vs_score": {"$meta": "searchScore"}}},
        {
            "$project": {
                "vs_score": {"$multiply": ["$vs_score", vector_scalar / vector_normalization]},
                "_id": 1,
                "raw": 1,
            }
        },
        {
            "$unionWith": {
                "coll": "sphere1mm",
                "pipeline": [
                    {
                        "$search": {
                            "index": "fts_sphere",
                            "text": {"query": query, "path": "raw"},
                        }
                    },
                    {"$limit": k * 2},
                    {"$addFields": {"fts_score": {"$meta": "searchScore"}}},
                    {
                        "$project": {
                            "fts_score": {"$multiply": ["$fts_score", fts_scalar / fts_normalization]},
                            "_id": 1,
                            "raw": 1,
                        }
                    },
                ],
            }
        },
        {
            "$group": {
                "_id": "$raw",
                "vs_score": {"$max": "$vs_score"},
                "fts_score": {"$max": "$fts_score"},
            }
        },
        {
            "$project": {
                "_id": 1,
                "raw": 1,
                "vs_score": {"$ifNull": ["$vs_score", 0]},
                "fts_score": {"$ifNull": ["$fts_score", 0]},
            }
        },
        {
            "$project": {
                "raw": 1,
                "score": {"$add": ["$fts_score", "$vs_score"]},
                "_id": 1,
                "vs_score": 1,
                "fts_score": 1,
            }
        },
        {"$limit": k},
        {"$sort": {"score": -1}},
    ]
```

**Benefits:**

This relative score fusion method offers several advantages:

- **Improved Search Relevance:** By combining vector search and full-text search, the results capture both semantic similarity and keyword relevance, leading to more accurate and comprehensive answers.
- **Flexibility:** The scaling factors for each score can be adjusted to prioritize either vector search or full-text search based on the specific needs and data characteristics.
- **Scalability:** The aggregation framework allows for efficient execution of the search queries even for large datasets.

**Future Directions:**

This work opens up exciting possibilities for further exploration:

- Investigating different score fusion techniques and weighting schemes.
- Integrating the approach with other search methods, such as entity search.
- Adapting the method to different datasets and NLP applications.

By leveraging relative score fusion, we can unlock the potential of hybrid search for enhanced information retrieval and deeper understanding of complex queries.
