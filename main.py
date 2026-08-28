import os
from fastapi import FastAPI
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.schema import QueryBundle
from fastapi.responses import StreamingResponse
from llama_index.llms.groq import Groq
from pydantic import BaseModel
import qdrant_client
from dotenv import load_dotenv
load_dotenv()

cohere_api_key = os.environ["COHERE_API"]

app = FastAPI()

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

collection_name = "enterprise_docs_v2"

vector_store = QdrantVectorStore(client=client, collection_name=collection_name, enable_hybrid=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
)

cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=10)


@app.post('/chat')
def chat(request: ChatRequest):
    filters = MetadataFilters(
    filters=[MetadataFilter(key="tenant_id", value=request.tenant_id)]
    )
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
            yield chunk.delta
    
    return StreamingResponse(stream_generator())


@app.get('/')
def api_status():
    return {"status": "API is running!"}