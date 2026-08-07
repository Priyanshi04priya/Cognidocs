"""Celery application — Redis is the broker and result backend."""

from celery import Celery

from app.config import settings

celery_app = Celery(
    "doc_query_engine",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_expires=3600 * 24,  # keep results 24h
    broker_connection_retry_on_startup=True,
    # Fail fast if Redis is down (avoids 60s Streamlit read timeouts)
    broker_transport_options={
        "socket_timeout": 5,
        "socket_connect_timeout": 5,
    },
    redis_socket_timeout=5,
    redis_socket_connect_timeout=5,
    result_backend_transport_options={
        "socket_timeout": 5,
        "socket_connect_timeout": 5,
    },
)
