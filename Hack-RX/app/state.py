from typing import TypedDict, Optional, List, Dict
from langchain_core.documents import Document
class AppState(TypedDict):
    jobId: str
    document_url: str
    original_questions: List[str]
    
    split_docs: Optional[List[Document]]
    ingestion_status: Optional[str]
    
    analyzed_queries: Optional[List[Dict]]
    shared_context: Optional[List[str]]
    reranked_contexts: Optional[List[List[str]]]
    final_answers: Optional[List[Dict]]



 