import os
from celery import Celery
from app.services.graph_service import execute_graph
import asyncio
from typing import List

redis_url = os.getenv("REDIS_URL")

celery_app = Celery(
    "tasks",
    broker=redis_url,
    backend=redis_url
)

@celery_app.task(name="process_document")
def process_document(jobId: str, url: str, questions: List[str]) -> dict:
    return asyncio.run(execute_graph(jobId, url, questions))