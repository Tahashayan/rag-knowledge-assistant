import os
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.postprocessor.cohere_rerank import CohereRerank
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from llama_index.core.schema import QueryBundle
from fastapi.responses import StreamingResponse
from langfuse import Langfuse
from fastapi.middleware.cors import CORSMiddleware
from llama_index.llms.groq import Groq
from pydantic import BaseModel
import qdrant_client
from dotenv import load_dotenv
load_dotenv()

LlamaIndexInstrumentor().instrument()

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
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
langfuse = Langfuse()

if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
    
LlamaIndexInstrumentor().instrument()

class ChatRequest(BaseModel):
    question : str
    tenant_id : str

Settings.llm = Groq(model="openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"])

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333,
    grpc_port=6334,
    prefer_grpc=True,  
    timeout=600,        
)

vector_store = QdrantVectorStore(client=client, collection_name="enterprise_docs_v2", enable_hybrid=True)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
cohere_rerank = CohereRerank(api_key=os.environ["COHERE_API"], top_n=10)


@app.post('/chat')
def chat(request: ChatRequest, api_key: str = Depends(get_api_key)):
    filters = MetadataFilters(filters=[MetadataFilter(key="tenant_id", value=request.tenant_id)])
    query_engine = index.as_retriever(filters=filters, similarity_top_k=20)
    
    nodes = query_engine.retrieve(request.question)
    
    reranked_nodes = cohere_rerank.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=request.question))
    
    if not reranked_nodes or reranked_nodes[0].score < 0.3:
        def fallback_generator():
            yield "I'm sorry, I don't have enough information in the company documents to answer that."
        return StreamingResponse(fallback_generator())
        
    context_str = ""
    for node in reranked_nodes:
        context_str += f"File Name: {node.metadata.get('file_name')}\nContent: {node.text}\n\n"
        
        
    prompt = f"""You are a helpful corporate assistant. Use ONLY the following context to answer the user's question. If the answer is not in the context, say 'I do not know'. At the end of your answer, cite the exact Source filenames you used.
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