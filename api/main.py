import os
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from llama_index.core import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.postprocessor.cohere_rerank import CohereRerank
from typing import List, Dict
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from llama_index.core.schema import QueryBundle
from fastapi.responses import StreamingResponse
from langfuse import Langfuse
from fastapi.middleware.cors import CORSMiddleware
from llama_index.llms.groq import Groq
from llama_index.embeddings.cohere import CohereEmbedding # <--- COHERE EMBEDDINGS
from pydantic import BaseModel
import qdrant_client
from dotenv import load_dotenv

load_dotenv()

LlamaIndexInstrumentor().instrument()
langfuse = Langfuse()

API_KEY = os.environ.get("MY_API_TOKEN", "default_dev_token") 
api_key_header = APIKeyHeader(name="Authorization", auto_error=True)

def get_api_key(api_key_header: str = Security(api_key_header)):
    token = api_key_header.replace("Bearer ", "") if "Bearer " in api_key_header else api_key_header
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API Key")
    return token

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://rag-knowledge-assistant-gamma.vercel.app"
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    tenant_id: str
    history: List[Dict[str, str]] = []

Settings.llm = Groq(model="openai/gpt-oss-120b", api_key=os.environ.get("GROQ_API_KEY"))
Settings.embed_model = CohereEmbedding(
    cohere_api_key=os.environ.get("COHERE_API"), 
    model_name="embed-english-v3.0",
    input_type="search_query"
)

client = qdrant_client.QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
    timeout=600
)
vector_store = QdrantVectorStore(client=client, collection_name="enterprise_docs_v2") # Hybrid removed for free tier
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
cohere_rerank = CohereRerank(api_key=os.environ.get("COHERE_API"), top_n=10)

@app.post('/chat')
def chat(request: ChatRequest, api_key: str = Depends(get_api_key)):
    
    history_str = ""
    if request.history:
        for msg in request.history:
            history_str += f"{msg['role'].upper()}: {msg['content']}\n"

    search_query = request.question
    if history_str:
        rewrite_prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question. 
        If the follow-up question is already standalone, just return it exactly as it is. 
        Do NOT answer the question, ONLY return the rewritten question.
        
        History:
        {history_str}
        
        Follow-up: {request.question}
        Standalone question:"""
        search_query = Settings.llm.complete(rewrite_prompt).text.strip()

    filters = MetadataFilters(filters=[MetadataFilter(key="tenant_id", value=request.tenant_id)])
    query_engine = index.as_retriever(filters=filters, similarity_top_k=20)
    
    nodes = query_engine.retrieve(search_query)
    reranked_nodes = cohere_rerank.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=search_query))

    if not reranked_nodes or reranked_nodes[0].score < 0.3:
        def fallback_generator():
            yield "I'm sorry, I don't have enough information in the company documents to answer that."
        return StreamingResponse(fallback_generator())

    context_str = ""
    for node in reranked_nodes:
        context_str += f"File Name: {node.metadata.get('file_name')}\nContent: {node.text}\n\n"
        
    prompt = f"""You are a helpful corporate assistant. Use ONLY the following context to answer the user's question. If the answer is not in the context, say 'I do not know'. At the end of your answer, cite the exact Source filenames you used.
    
    CHAT HISTORY:
    {history_str}
    
    CONTEXT: 
    {context_str}
    
    QUESTION: 
    {request.question} 
    ANSWER:
    """
    
    def stream_generator():
        response = Settings.llm.stream_complete(prompt)
        for chunk in response:
            if chunk.delta:
                yield chunk.delta

    return StreamingResponse(stream_generator())

@app.get('/')
def api_status():
    return {"status": "API is running!"}