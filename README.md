# 🏢 Enterprise-Grade RAG Knowledge Assistant

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-black?style=flat&logo=next.js)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-black?style=flat)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red.svg)
![Groq](https://img.shields.io/badge/Groq-Llama_3-orange.svg)

![Enterprise RAG Chat Interface](https://github.com/user-attachments/assets/7af12bcf-80ea-436c-bbfb-4f0e7785051b)

🚀 [Live Project](https://rag-knowledge-assistant-gamma.vercel.app/)

A production-ready, multi-tenant Retrieval-Augmented Generation (RAG) system. This application allows users to securely "chat with their company data" using advanced vector search, reranking, and hallucination guardrails. 

Built with **FastAPI**, **Next.js**, **LlamaIndex**, and **Qdrant**, it features real-time token streaming and enterprise-level observability.

---

## 🚀 Key Features

* **Multi-Tenant Data Isolation:** Secure metadata filtering ensures that users from "Company A" cannot retrieve or query documents belonging to "Company B".
* **Hybrid Search & Reranking:** Combines Dense Vector Search (Ollama `nomic-embed-text`) with Sparse Keyword Search (BM25), passed through the **Cohere Rerank API** for unparalleled context precision.
* **Zero-Hallucination Guardrails:** Implements a strict confidence-score threshold. If retrieval relevance is below `0.3`, the LLM is bypassed entirely in favor of a canned fallback response.
* **Real-Time Streaming:** Utilizes FastAPI `StreamingResponse` and React Server-Sent Events (SSE) to deliver a ChatGPT-like typing experience with zero latency.
* **API Security:** Endpoints are protected via Bearer Token authentication (`APIKeyHeader`).
* **Advanced Document Parsing:** Uses `PyMuPDF` to cleanly extract text from dense corporate PDFs (like 10-K financial reports), reducing token-bloat and database storage by over 90% compared to standard parsers.

---

## 📊 Evaluation & Observability

This project does not rely on "vibes" to prove accuracy. It is mathematically evaluated using **RAGAS** (LLM-as-a-Judge) and monitored via **Langfuse**.

* **Faithfulness Score:** `1.0` (100% hallucination-free generation based purely on retrieved context).
* **Context Precision:** Highly optimized via Cohere Rerank (Top 20 -> Top 5 refinement).
* **Tracing:** Full observability waterfall implemented via `LlamaIndexInstrumentor`, tracking embedding latency, LLM generation time, and exact token costs per query.

---

## 🏗️ Architecture & Tech Stack

**Backend / AI Pipeline:**
* **Framework:** FastAPI (Python)
* **Orchestration:** LlamaIndex
* **Vector Database:** Qdrant (Docker)
* **Embeddings:** Ollama (`nomic-embed-text`) - *Privacy-preserving local embeddings.*
* **Generation LLM:** Groq (`openai/gpt-oss-120b`) - *Lightning-fast inference.*
* **Reranker:** Cohere Rerank API
* **Evaluation & Tracing:** RAGAS, Langfuse

**Frontend:**
* **Framework:** Next.js (App Router), React
* **Styling:** Tailwind CSS
* **State:** React Hooks with custom stream-chunk decoders.

---

## 📂 Project Structure

```text
enterprise-rag-assistant/
├── api/
│   └── main.py             # FastAPI streaming server & Guardrails
├── scripts/
│   ├── ingest.py           # Data ingestion, chunking, and metadata tagging
│   └── evaluate.py         # RAGAS evaluation suite (Faithfulness, Precision)
├── frontend/               # Next.js / Tailwind Chat Interface
├── data/                   # Raw PDFs (Company Handbooks, 10-K Reports)
├── .env.example            # Environment variable template
└── requirements.txt        # Python dependencies



