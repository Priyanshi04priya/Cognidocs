from app.state import AppState
import asyncio
from app.services.llm_service import (
    analyze_query,
    rerank_documents,
    generate_initial_decision,
)
from app.components.data_ingestion import result
from app.components.data_preproceesing import load_document
from app.services.retrival import retrieve_from_qdrant

async def preprocessing_node(state: AppState) -> AppState:
    print(f"---NODE (Job ID: {state['jobId']}): PREPROCESSING DOCUMENT---")
    split_docs = load_document(state['document_url'])
    print(f"Successfully split document into {len(split_docs)} chunks.")
    return {"split_docs": split_docs}

async def db_loading_node(state: AppState) -> AppState:
    collection_name = state['jobId']
    print(f"---NODE (Job ID: {collection_name}): LOADING CHUNKS TO DB---")
    ingestion_status = await asyncio.to_thread(result, state['split_docs'], collection_name)
    return {"ingestion_status": ingestion_status}

async def query_analysis_node(state: AppState) -> AppState:
    questions = state['original_questions']
    print(f"---NODE (Job ID: {state['jobId']}): BATCH ANALYZING {len(questions)} QUERIES---")
    analysis_tasks = [analyze_query(q) for q in questions]
    analyzed_queries = await asyncio.gather(*analysis_tasks)
    
    primary_domain = analyzed_queries[0].get("domain", "general") if analyzed_queries else "general"
    for doc in state['split_docs']:
        doc.metadata['domain'] = primary_domain
        
    return {"analyzed_queries": analyzed_queries}

async def wait_for_indexing_node(state: AppState) -> AppState:
    wait_seconds = 1
    print(f"---NODE (Job ID: {state['jobId']}): WAITING FOR {wait_seconds}s FOR DB INDEXING---")
    await asyncio.sleep(wait_seconds)
    return {}

async def retrieval_node(state: AppState) -> AppState:
    collection_name = state['jobId']
    analyzed_queries = state['analyzed_queries']
    print(f"---NODE (Job ID: {collection_name}): BATCH RETRIEVING DOCUMENTS---")
    
    all_search_queries = [sq for analysis in analyzed_queries for sq in analysis.get("search_queries", [])]
    primary_domain = analyzed_queries[0].get("domain") if analyzed_queries else "general"
    
    retrieved_docs = await retrieve_from_qdrant(all_search_queries, primary_domain, collection_name)
    return {"shared_context": retrieved_docs}

async def rerank_node(state: AppState) -> AppState:
    
    original_questions = state['original_questions']
    shared_context = state['shared_context']
    print(f"---NODE (Job ID: {state['jobId']}): BATCH RERANKING CONTEXT FOR {len(original_questions)} QUERIES---")

    rerank_tasks = [rerank_documents(query, shared_context) for query in original_questions]
    reranked_contexts = await asyncio.gather(*rerank_tasks)

    return {"reranked_contexts": reranked_contexts}

async def generation_node(state: AppState) -> AppState:
    
    analyzed_queries = state['analyzed_queries']
    reranked_contexts = state['reranked_contexts']
    print(f"---NODE (Job ID: {state['jobId']}): BATCH GENERATING {len(analyzed_queries)} ANSWERS---")
    
    generation_tasks = [
        generate_initial_decision(analyzed_queries[i], reranked_contexts[i])
        for i in range(len(analyzed_queries))
    ]
    results_with_critiques = await asyncio.gather(*generation_tasks)
    
    final_answers = [result[0] for result in results_with_critiques]
    # critiques = [result[1] for result in results_with_critiques]
    
    state["final_answers"] = final_answers
    return state

async def correction_node(state: AppState) -> AppState:
    """Prepares feedback for a correction attempt."""
    print("---NODE: PREPARING CORRECTION---")
    critique = state.get("critique", {})
    state["correction_feedback"] = critique
    print(f"Correction feedback: {critique.get('feedback')}")
    state["needs_correction"] = False 
    return state
