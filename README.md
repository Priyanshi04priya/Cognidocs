## AI-Powered Document Query Engine

A domain-agnostic intelligent document analysis system built using **FastAPI, LangGraph, and Qdrant**.  
Upload documents, submit multiple natural-language questions, and receive structured, verifiable answers — all through a single API request.

---

## ✨ Key Features

- **📍 Unified Processing Endpoint** – A single `/process` API handles both document ingestion and batch question answering in one request.
- **🌎 Domain-Independent** – Automatically detects the document’s domain (Insurance, Legal, HR, Finance, etc.) and extracts relevant entities accordingly.
- **🔒 Isolated Vector Storage** – Each job runs in its own Qdrant collection (`jobId`) to ensure strict data separation.
- **⚡ Concurrent Query Handling** – Processes multiple questions in parallel for faster response times.

### 🧠 Advanced RAG Pipeline

- **Retrieval:** Fetches broadly relevant document chunks  
- **Reranking:** Uses a CrossEncoder to refine context  
- **Generation:** A dual-persona LLM (Analyst + Auditor) produces structured answers with self-evaluation and confidence scoring

- **📂 Multi-Format Compatibility** – Supports PDF, DOCX, and EML documents.

---

## 🏗 Architecture Overview

Built on a **LangGraph stateful workflow** for transparency and maintainability:

- **FastAPI Server** — Accepts requests and coordinates processing
- **LangGraph Workflow** — Executes a sequence of debuggable processing nodes
- **Qdrant Vector Database** — Stores embeddings per job
- **LangChain + OpenAI** — Powers query understanding and answer generation
- **Sentence Transformers** — Improves relevance through reranking
- **Document Parsers** — PyMuPDF, python-docx, and mailparser extract clean text

---

## 🔍 Processing Workflow

Calling the `/process` endpoint triggers the pipeline:

1. **preprocess** — Download, parse, and chunk the document  
2. **batch_analyze_queries** — Detect domain and craft search queries  
3. **load_to_db** — Store chunks in a job-specific vector collection  
4. **batch_retrieve_docs** — Gather relevant context  
5. **batch_rerank_docs** — Produce question-specific context  
6. **batch_generate_answers** — Generate structured answers with self-review  

---

## 📡 API Usage

### POST `/process`

Upload a document, process questions, and receive structured responses.

### Request

```json
{
  "jobId": "string",
  "documents": "string (URL)",
  "questions": ["string"]
}
```
- **jobId** — Unique identifier used as the Qdrant collection name  
- **documents** — Public URL to a PDF/DOCX/EML file  
- **questions** — List of natural language queries
### Response
```json
{
  "answers": [
    {
      "decision": "string",
      "details": {},
      "justification": "string",
      "clauses": ["string"]
    }
  ]
}
```
- **decision** — Final concise outcome (e.g., “Approved”)
- **details** — Extracted key information
- **justification** — Step-by-step reasoning grounded in the document
- **clauses** — Supporting excerpts or references
## ⚙️ Setup Instructions

Follow these steps to set up and run the project locally.

---

### 📌 Prerequisites

- Python 3.9+
- Docker (for Qdrant)
- OpenAI API Key

---
## 📥 Installation

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```
### 🔐 Environment Configuration

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY="sk-..."
QDRANT_URL="http://localhost:6333"
QDRANT_API_KEY=null
EMBEDDING_MODEL="text-embedding-3-small"
```
### 🐳 Run Qdrant (Vector Database)

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```
## ▶️ Running the Application

```bash
uvicorn main:app --reload
```
### Access the Application

API: http://127.0.0.1:8000

Interactive Docs (Swagger): http://127.0.0.1:8000/docs
## 💻 Technology Stack

| Component     | Technology                       |
|---------------|----------------------------------|
| Backend       | FastAPI                          |
| Workflow      | LangGraph                        |
| RAG Framework | LangChain                        |
| Vector DB     | Qdrant                           |
| LLM           | OpenAI                           |
| Reranker      | Sentence Transformers            |
| Parsing       | PyMuPDF, python-docx, mailparser |

## 🏆 Why This Project Matters

This system goes beyond a typical Q&A tool. It is a **scalable, explainable, and high-accuracy document intelligence platform** capable of adapting to diverse industries while delivering trustworthy, auditable answers at scale.
