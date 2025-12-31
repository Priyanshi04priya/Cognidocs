from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List, Optional, Dict, Any,Tuple
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from fastapi import HTTPException
from openai import OpenAI
import os

load_dotenv()


class AnalyzedQuery(BaseModel):
    domain: str =Field(..., description="The general domain of the query (e.g., 'Insurance', 'Legal Compliance').")
    key_entities: Optional[Dict[str, Any]] = Field(default_factory=dict, description="A dictionary of key-value pairs extracted from the query.")
    search_queries: List[str] = Field(..., description="A list of 3-5 detailed questions for semantic search.")
    hypotheses: List[str] = Field(..., description="A list of potential outcomes or answers.")

class DocumentQueryResult(BaseModel):
    decision: str = Field(..., description="The final, summary answer to the user's query.")
    details: Dict[str, Any] = Field(default_factory=dict, description="A dictionary containing specific results. Give any amount (money) details if available.")
    justification: str = Field(..., description="A clear, step-by-step reasoning for the decision.")
    clauses: List[str] = Field(default_factory=list, description="A list of specific clause IDs or text snippets from the documents.")

class DecisionCritique(BaseModel):
    correction_needed: bool = Field(..., description="Whether the original decision needs to be corrected.")
    confidence_score: float = Field(..., description="A score from 0.0 to 1.0 indicating confidence.")
    feedback: str = Field(..., description="Detailed feedback for why a correction is or is not needed.")

class CombinedResponse(BaseModel):
    decision: DocumentQueryResult
    critique: DecisionCritique


try:
    llm=ChatOpenAI(temperature=0, model="gpt-4.1-mini",
    max_tokens=1024,
    )
    # llm = OpenAI(
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    # )
    # cannot get structured output using openai()-> 
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
except Exception as e:
    print(f"CRITICAL: Failed to initialize AI models: {e}")
    llm = None
    cross_encoder = None
async def analyze_query(query: str) -> dict:
    try:
        SYSTEM_PROMPT = """
        You are a Domain-Aware Query Analyzer Assistant. Your job is to convert unstructured user queries into structured representations that downstream applications can use for accurate document retrieval, policy validation, contract management, etc.

        You MUST:
        1. Identify the domain of the query (e.g., Insurance, Legal Compliance, Human Resources, Contract Management).
        2. Extract key entities relevant to that domain in a structured dictionary format.
        3. Generate 3-5 diverse but relevant search queries that capture the core intent of the original query.
        4. Generate 2-4 likely hypotheses — assumptions or possible interpretations that the system might make **before** retrieving documents.

        Guidelines:
        - Tailor extracted entities and hypotheses to the identified domain.
        - Combine extractive and generative reasoning: Extract key facts AND generate potential queries and interpretations.
        - Handle ambiguity gracefully. If uncertain, make reasonable guesses and reflect them in the hypotheses.

        Output Schema (Python Pydantic):
        - domain: str — e.g., "Insurance"
        - key_entities: dict[str, str] — e.g., {{"age": "46", "procedure": "knee surgery", "location": "Pune"}}
        - search_queries: List[str] — e.g., ["Is knee surgery covered in 3-month policy?", ...]
        - hypotheses: List[str] — e.g., ["Waiting period might not be over", ...]

        Example Query:
        "46M, knee surgery, Pune, 3-month policy"

        Example Output:
        {{
        "domain": "Insurance",
        "key_entities": {{
            "age": "46M",
            "procedure": "knee surgery",
            "location": "Pune",
            "policy_duration": "3 months"
        }},
        "search_queries": [
            "Is knee surgery covered under a 3-month policy?",
            "Waiting period for knee surgery in health insurance",
            "Does a 3-month-old policy cover planned hospitalization?",
            "Coverage for knee surgery in Pune hospitals",
            "Knee surgery eligibility in short-term insurance plans"
        ],
        "hypotheses": [
            "The procedure may not be covered due to a waiting period.",
            "User is trying to check claim eligibility for upcoming surgery.",
            "Knee issue could be pre-existing and may affect approval.",
            "User likely seeks cashless hospital options in Pune."
        ]
        }}
        """
        prompt=ChatPromptTemplate.from_messages(
            [
                    ("system", SYSTEM_PROMPT),
                    ("human", "Translate the following query into a structured JSON format: {query}"),
            ]
        )
        structured_query = llm.with_structured_output(AnalyzedQuery, method="function_calling")
        chain=prompt | structured_query
        response=  await chain.ainvoke({"query": query})
        return response.model_dump()
    except Exception as e:
        print("Error in analyze_query:", e)
        raise HTTPException(status_code=500, detail=str(e))
async def generate_initial_decision(analyzed_query: dict, docs: list, feedback: Optional[dict] = None) -> Tuple[dict, dict]:
    context = "\n\n".join(doc for doc in docs)
    correction_instruction=None
    if feedback:
        correction_instruction=f"""this is the second attempt, firts attempt was flawed Plz pay close attention to the feedback
        {feedback.get("feedback")}
        """
    system_prompt=ChatPromptTemplate.from_messages(
        [("system",
         """
        you are multi-persona AI assistant.you will act as a meticulous Analyst for the {domain} domain and then as a skeptical Senior Auditor.
        
        Analyzed Query: {analyzed_query}
        Context: {context}
        Correction Instruction: {correction_instruction}
        -----------------------------------------------------
        PART 1: Analyst's Decision
        1. Fact Check: Use the `key_entities` from the Analyzed Query as your source of truth.
        2.  Clause Matching: For each key entity, find the most relevant document clause(s).
        3. Step-by-Step Reasoning: Evaluate each entity against its matched clause. Your reasoning must be specific to the {domain}domain.
        4. Synthesize Decision: Combine your evaluations into a final decision.
        5. Format Output: Provide your final decision as a JSON object conforming to the `DocumentQueryResult` schema. The `decision` and `details` fields must be relevant to the domain.
        6. try to keep the ans as short as possible and to the point. No extra information.
        ----------------------------------------------------------
        Part 2: Auditor's Critique
        
        Now, as the Senior Auditor, critically evaluate the Analyst's decision.
        1. Fact Verification: Did the Analyst correctly use all `key_entities`? Were any clauses misinterpreted?
        2. Logical Soundness: Is the `justification` logical and does it directly lead to the `decision`?
        3. Format Critique: Provide your critique as a JSON object conforming to the `DecisionCritique` schema.

        ----------------------------------------------------------

        Combined Final Output:
        Return a single JSON object with two keys: "decision" (the `DocumentQueryResult` JSON) and "critique" (the `DecisionCritique` JSON).
        """
        ),
        ("human", "Analyzed Query: \n{analyzed_query}\n\nContext:\n{context}")
        ]
    )
    
    structured_llm = llm.with_structured_output(CombinedResponse)
    chain = system_prompt | structured_llm
    
    try:
        response = await chain.ainvoke({
            "domain": analyzed_query.get("domain"),
            "analyzed_query": analyzed_query,
            "context": context,
            "correction_instruction": correction_instruction
        })
        return response.decision.model_dump(), response.critique.model_dump()
    except Exception as e:
        print(f"Error in generate_initial_decision: {e}")
        error_decision = {
            "decision": "Error", 
            "details": {}, 
            "justification": f"Failed to generate a decision due to an internal error: {e}", 
            "clauses": []
        }
        error_critique = {
            "correction_needed": False, 
            "confidence_score": 0.0, 
            "feedback": "Generation failed."
        }
        return error_decision, error_critique

async def rerank_documents(original_query: str, documents: List[str]) -> List[str]:
    if not documents:
        return []
    query_doc_pairs = [[original_query, doc] for doc in documents]
    
    scores = cross_encoder.predict(query_doc_pairs)
    
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    reranked_docs = [doc for doc, score in doc_scores]
    
    return reranked_docs
   
    