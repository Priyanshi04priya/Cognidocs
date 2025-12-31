# from fastapi import FastAPI,HTTPException
# from pydantic import BaseModel
# from typing import Optional,List,Dict
# from app.services.graph_service import execute_graph
# from app.celery_worker import process_document
# from celery.result import AsyncResult


# app = FastAPI(
#     title="Insurance Claim Analysis",
#     description="API for Insurance Claim Analysis",
# )
# @app.get("/")
# def health_check():
#     return {"message":"server is running"}

# class QueryRequest(BaseModel):
#     jobId:str
#     # url of the document
#     documents:str
#     questions: List[str]

# class DecisionResponse(BaseModel):
#     decision: str
#     amount: Optional[float] = None
#     justification: str
#     clauses: list[str]

# class SubmitResponse(BaseModel):
#     task_id: str
#     status: str
#     message: str

# class ProcessResponse(BaseModel):
#     answers: List[DecisionResponse]
    
# class StatusResponse(BaseModel):
#     task_id: str
#     status: str
#     result: Optional[Dict[str, List[DecisionResponse]]] = None

    
# @app.post("/process", response_model=ProcessResponse)
# async def process_document_and_query(request: QueryRequest):
#     if not all([request.jobId, request.documents, request.questions]):
#         raise HTTPException(status_code=400, detail="jobId, documents (url), and a list of questions are required.")

#     try:
#         results = await execute_graph(
#             request.jobId, request.documents, request.questions
#         )
#         return {"answers": results}
#     except Exception as e:
#         print(f"An error occurred during graph execution for jobId {request.jobId}: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# # @app.get("/status/{task_id}", response_model=StatusResponse)
# # def get_job_status(task_id: str):
# #     task_result = AsyncResult(task_id, app=process_document.app)

# #     if not task_result:
# #         raise HTTPException(status_code=404, detail="Task not found.")

# #     if task_result.ready():
# #         if task_result.successful():
# #             return {
# #                 "task_id": task_id,
# #                 "status": "SUCCESS",
# #                 "result": task_result.get()
# #             }
# #         else: 
# #             return {
# #                 "task_id": task_id,
# #                 "status": "FAILURE",
# #                 "result": {"error": str(task_result.info)}
# #             }
# #     return {"task_id": task_id, "status": "PENDING", "result": None}

