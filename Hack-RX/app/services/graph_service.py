from langgraph.graph import StateGraph, END
from app.services.nodes import (
    preprocessing_node,
    wait_for_indexing_node,
    db_loading_node,
    query_analysis_node,
    retrieval_node,
    rerank_node,
    generation_node,
    correction_node,
)
from typing import List
from app.state import AppState

def should_correct(state: AppState) -> str:
    if state.get("needs_correction"):
        print("---DECISION: CORRECTION NEEDED---")
        return "correct"
    else :
        print("---DECISION: NO CORRECTION NEEDED---")
        return END
async def execute_graph(jobId: str, url: str, questions: List[str]) -> List[dict]:
    workflow = StateGraph(AppState)
    
    workflow.add_node("preprocessing", preprocessing_node)
    workflow.add_node("load_to_db",db_loading_node)
    workflow.add_node("wait_for_indexing", wait_for_indexing_node)

    workflow.add_node("query_analysis",query_analysis_node)
    workflow.add_node("retrieve",retrieval_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generation_node)
    workflow.add_node("correct", correction_node)

    workflow.set_entry_point("preprocessing")
    workflow.add_edge("preprocessing", "query_analysis")
    workflow.add_edge("query_analysis", "load_to_db")
    workflow.add_edge("load_to_db", "wait_for_indexing")
    
    workflow.add_edge("wait_for_indexing", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)
    
    # workflow.add_conditional_edges(
    # "generate",
    # {"correct": "correct", END:END},
    # should_correct
    # )
    


    app_graph = workflow.compile()

    initial_state = {
        "jobId": jobId,
        "document_url": url,
        "original_questions": questions,
    }
    final_state = await app_graph.ainvoke(initial_state)
    if not final_state.get("final_answers"):
        raise Exception("Graph execution failed to produce a final response.")

    return final_state["final_answers"]

# questions=[
#         "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
#         "What is the waiting period for pre-existing diseases (PED) to be covered?"
#   ]
# asyncio.run(execute_graph("1","https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            #   questions
              
# ))