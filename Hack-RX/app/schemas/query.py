from pydantic import BaseModel
from typing import List
class QueryRequest(BaseModel):
    # pydantic model for user query
    query:str

class QueryResponse(BaseModel):
    # pydantic model for llm response sent to user 
    # a json or a string
    answer:str
    source_chunks:List[str]
    pass
    