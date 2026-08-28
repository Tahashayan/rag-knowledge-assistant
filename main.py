import os
from fastapi import FastAPI
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.postprocessor.cohere_rerank import CohereRerank
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from llama_index.core.schema import QueryBundle
from fastapi.responses import StreamingResponse
from langfuse import get_client
from llama_index.llms.groq import Groq
from pydantic import BaseModel
import qdrant_client
from dotenv import load_dotenv
load_dotenv()

cohere_api_key = os.environ["COHERE_API"]
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-...")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-...")
os.environ.setdefault("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

app = FastAPI()
langfuse = get_client()

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
    with langfuse.start_as_current_observation(
        as_type="span",
        name="enterprise-rag-chat",
    ) as trace:
        trace.update(
            input={
                "question": request.question,
                "tenant_id": request.tenant_id
            }
        )
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
        full_response = ""
        with langfuse.start_as_current_observation(
        as_type="generation",
        name="groq-rag-generation"
        ) as generation:
            generation.update(
            input={
                "question": request.question,
                "prompt": prompt
            },
            model="openai/gpt-oss-120b"
        )
        response = Settings.llm.stream_complete(prompt)
        for chunk in response:
            if chunk.delta:
                full_response += chunk.delta
                yield chunk.delta
        generation.update(
            output=full_response
        )
        trace.update(
            output={
                "answer": full_response
            }
        )
    langfuse.flush()
    return StreamingResponse(stream_generator())


@app.get('/')
def api_status():
    return {"status": "API is running!"}