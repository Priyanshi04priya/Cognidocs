from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_community.embeddings import OpenAIEmbeddings
from qdrant_client import AsyncQdrantClient, models
import os
import asyncio

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))

try:
    qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
except Exception as e:
    print(f"CRITICAL: Failed to initialize Qdrant client or Embedding model: {e}")
    qdrant_client = None
    embedding_model = None


async def _search_single_query(query: str, domain: str, collection_name: str) -> List[models.ScoredPoint]:
    try:
        query_vector = await embedding_model.aembed_query(query)
        search_results = await qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.domain",
                        match=models.MatchValue(value=domain),
                    )
                ]
            ),
            limit=SEARCH_TOP_K,
            with_payload=True,
        )
        return search_results
    except Exception as e:
        print(f"Error searching for query '{query}': {e}")
        return []


async def retrieve_from_qdrant(search_queries: List[str], domain: str, collection_name: str) -> List[str]:
    if not qdrant_client or not embedding_model:
        print("ERROR: Qdrant client/embedding model not initialized. Cannot retrieve documents.")
        return []

    if not search_queries:
        print("No search queries provided")
        return []

    print(f"Retrieving documents from Qdrant collection '{collection_name}' for {len(search_queries)} search queries")

    search_tasks = [asyncio.create_task(_search_single_query(q, domain, collection_name)) for q in search_queries]
    list_of_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    
    unique_documents = {}
    for result in list_of_results:
        if isinstance(result, Exception):
            print(f"A search task failed: {result}")
            continue
        for scored_point in result:
            
            if scored_point.id not in unique_documents:
                
                unique_documents[scored_point.id] = {
                    "content": scored_point.payload.get("content", ""),
                    "score": scored_point.score
                }

    
    sorted_docs = sorted(unique_documents.values(), key=lambda x: x["score"], reverse=True)
    final_content = [doc["content"] for doc in sorted_docs]
    
    print(f"Retrieved {len(final_content)} unique documents after deduplication")
    return final_content

