from app.services.nodes import *
nodes_list = [
    preprocessing_node,
    wait_for_indexing_node,
    db_loading_node,
    query_analysis_node,
    retrieval_node,
    rerank_node,
    generation_node,
    correction_node
]

for idx, node in enumerate(nodes_list, start=1):
    print(f"Node {idx} ({node.__name__ if callable(node) else type(node)}): {node}")
