from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field,RootModel
from langchain.chat_models import init_chat_model
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv
import asyncio
load_dotenv()
class KeyEntities(RootModel[dict[str, Union[str, int, float]]]):
    pass
        

class AnalyzedQuery(BaseModel):
    """Structured representation of a user's insurance query.""" 
    domain: str = Field(..., description="The general domain of the query (e.g., 'Insurance', 'Legal Compliance', 'Human Resources', 'Contract Management').")
    
    key_entities: Optional[KeyEntities] = Field(..., description="A dictionary of key-value pairs extracted from the query, specific to the identified domain.")
    
    search_queries: List[str] = Field(..., description="A list of 3-5 detailed, rephrased questions for semantic search against a document database, tailored to the domain.")
    
    hypotheses: List[str] = Field(..., description="A list of potential outcomes or answers based on the query and common rules within the domain.")

    
# Core LLM Functions
llm=init_chat_model(model_provider="openai",model="gpt-4.1")
async def analyze_query(query: str) -> dict:
    # LLM call to analyze query and translate to structured Json format
    prompt=ChatPromptTemplate.from_messages(
        [
            
        ("system", 
            "You are an expert at parsing and analyzing insurance claim queries. "
            "Your task is to convert a user's raw query into a structured JSON object. "
            "Based on the query, extract key entities. "
            "Generate 3-4 very specific questions to search a vector database of policy documents. These questions should seek to verify coverage, check waiting periods, confirm network status for locations, and identify exclusions related to the user's query. "
            "Also, generate hypotheses about potential outcomes based on common insurance rules."
        ),
        ("human", "Translate the following query into a structured JSON format: {query}"),
    
        ]
    )
    structured_query = llm.with_structured_output(AnalyzedQuery, method="function_calling")
    chain=prompt | structured_query
    response=  await chain.ainvoke({"query": query})
    print("\n\nResponse: ",response)
    
query="How do HR departments ensure compliance with maternity leave laws in India?"

print("Query: ",query)

asyncio.run(analyze_query(query))