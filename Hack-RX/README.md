
# AI-Powered Document Query Engine

An advanced **domain-agnostic document analysis system** built with **FastAPI**, **LangGraph**, and **Qdrant**.  
Easily ingest documents, ask multiple natural language questions, and receive **structured, verifiable answers** — all in **one API call**.

---

## ✨ Key Highlights

- **📍 One API to Rule Them All** – A single `/process` endpoint handles **document ingestion** + **batch querying** in one go.
- **🌎 Works Across Domains** – Insurance, Legal, HR, Finance — the system detects the domain and extracts relevant entities automatically.
- **🔒 Isolated Vector Collections** – Each job has its own Qdrant collection (`jobId`) for **data segregation**.
- **⚡ Parallel Question Processing** – Handles multiple queries simultaneously for **high throughput**.
- **🧠 Advanced RAG Pipeline**:
  1. **Retrieval** – Get broad, relevant document chunks.  
  2. **Reranking** – Refine with a **CrossEncoder** for precise context.  
  3. **Generation** – Multi-persona LLM (Analyst + Auditor) produces **structured answers** with **self-critique** + **confidence score**.
- **📂 Multi-Format Support** – Works with **PDF, DOCX, and EML** files.

---

## 🏗 Architecture

Powered by a **LangGraph**-based graph workflow:

1. **FastAPI Server** – Receives API requests and orchestrates the workflow.
2. **LangGraph Workflow** – A sequence of stateful nodes ensuring debuggable and maintainable execution.
3. **Qdrant Vector DB** – Stores vectorized document chunks per `jobId`.
4. **LangChain + OpenAI** – Handles question analysis, structured answer generation, and self-critique.
5. **Sentence Transformers** – Reranks retrieved documents for maximum relevance.
6. **Document Parsers** – PyMuPDF, python-docx, and mailparser extract clean text from various formats.

---

## 🔍 Workflow Overview

**The `/process` endpoint triggers this sequence:**

1. **`preprocess`** – Download, parse, and chunk the document.  
2. **`batch_analyze_queries`** – Identify document domain + generate search queries for each question.  
3. **`load_to_db`** – Store chunks in a **job-specific** Qdrant collection.  
4. **`batch_retrieve_docs`** – Retrieve broad context for all questions.  
5. **`batch_rerank_docs`** – Use CrossEncoder to create **tailored context per question**.  
6. **`batch_generate_answers`** – Generate structured answers + self-critique.

---

## 📡 API Usage

### **POST** `/process`

Ingest a document, process questions, and receive structured answers.

**Request Body:**
```json
{
  "jobId": "string",
  "documents": "string (URL)",
  "questions": ["string"]
}
````

* **jobId** – Unique identifier, used as Qdrant collection name.
* **documents** – Public URL to PDF/DOCX/EML file.
* **questions** – List of natural language questions.

**Response:**

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

* **decision** – Short, direct outcome (e.g., `"Approved"`).
* **details** – Extracted facts (e.g., amount, waiting period).
* **justification** – Step-by-step reasoning based on document.
* **clauses** – Supporting excerpts or clause IDs.

---

## ⚙️ Setup

### **Prerequisites**

* Python 3.9+
* Docker (for Qdrant)
* OpenAI API key

### **1. Clone the Repository**

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Configure Environment**

Create `.env`:

```env
OPENAI_API_KEY="sk-..."
QDRANT_URL="http://localhost:6333"
QDRANT_API_KEY=null
EMBEDDING_MODEL="text-embedding-3-small"
```

### **4. Run Qdrant (via Docker)**

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

---

## ▶️ Run the Application

```bash
uvicorn main:app --reload
```

Access:

* **API** – `http://127.0.0.1:8000`
* **Docs** – `http://127.0.0.1:8000/docs`

---

## 💻 Tech Stack

| Component              | Technology                           |
| ---------------------- | ------------------------------------ |
| Backend                | FastAPI                              |
| Workflow Orchestration | LangGraph                            |
| RAG/LLM Framework      | LangChain                            |
| Vector Database        | Qdrant                               |
| LLM Provider           | OpenAI                               |
| Reranker               | Sentence Transformers (CrossEncoder) |
| Document Parsing       | PyMuPDF, python-docx, mailparser     |

---

## 🏆 Why This Matters

This engine isn't just a Q\&A bot — it’s a **scalable, auditable, and high-precision** document understanding system that adapts to any industry, providing reliable and explainable answers at scale.
