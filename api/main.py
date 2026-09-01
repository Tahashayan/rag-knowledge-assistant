import os
from typing import List, Dict
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import qdrant_client
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter
)
from llama_index.core.schema import QueryBundle
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.llms.groq import Groq
from openinference.instrumentation.llama_index import (
    LlamaIndexInstrumentor
)
from langfuse import get_client
load_dotenv()
LlamaIndexInstrumentor().instrument()

API_KEY = os.environ.get(
    "MY_API_TOKEN",
    "default_dev_token"
)

api_key_header = APIKeyHeader(
    name="Authorization",
    auto_error=True
)


def get_api_key(
    api_key_header: str = Security(api_key_header)
):
    token = (
        api_key_header.replace("Bearer ", "")
        if "Bearer " in api_key_header
        else api_key_header
    )

    if token != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Could not validate API Key"
        )

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

langfuse = get_client()

if langfuse.auth_check():
    print(
        "Langfuse client is authenticated and ready!"
    )
else:
    print(
        "Authentication failed. "
        "Please check your credentials and host."
    )

class ChatRequest(BaseModel):
    question: str
    tenant_id: str
    history: List[Dict[str, str]] = []
    
Settings.llm = Groq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"]
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

client = qdrant_client.QdrantClient(
    url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
    api_key=os.environ.get("QDRANT_API_KEY", ""),
    port=6333,
    grpc_port=6334,
    prefer_grpc=True,
    timeout=600,
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="enterprise_docs_v2",
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store
)

cohere_rerank = CohereRerank(
    api_key=os.environ["COHERE_API"],
    top_n=10
)

@app.post("/chat")
def chat(
    request: ChatRequest,
    api_key: str = Depends(get_api_key)
):

    trace = langfuse.start_observation(
        name="corporate-chat",
        as_type="span",
        metadata={
            "tenant_id": request.tenant_id,
            "original_query": request.question
        }
    )

    history_str = ""

    if request.history:
        for msg in request.history:
            history_str += (
                f"{msg['role'].upper()}: "
                f"{msg['content']}\n"
            )
            
    search_query = request.question

    if history_str:

        rewrite_prompt = f"""
            Given the following conversation history and a follow-up question,
            rephrase the follow-up question to be a standalone question.

            If the follow-up question is already standalone, just return it
            exactly as it is.

            Do NOT answer the question.
            ONLY return the rewritten question.

            History:
            {history_str}

            Follow-up:
            {request.question}

            Standalone question:
            """

        # Fast non-streaming call for query rewriting
        search_query = Settings.llm.complete(
            rewrite_prompt
        ).text.strip()

        # Store rewritten query in Langfuse
        trace.update(
            metadata={
                "original_query": request.question,
                "rewritten_query": search_query,
                "history_present": True
            }
        )

    else:

        trace.update(
            metadata={
                "original_query": request.question,
                "rewritten_query": search_query,
                "history_present": False
            }
        )
        
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="tenant_id",
                value=request.tenant_id
            )
        ]
    )

    query_engine = index.as_retriever(
        filters=filters,
        similarity_top_k=20
    )

    nodes = query_engine.retrieve(
        search_query
    )

    reranked_nodes = cohere_rerank.postprocess_nodes(
        nodes,
        query_bundle=QueryBundle(
            query_str=search_query
        )
    )
    
    if (
        not reranked_nodes
        or reranked_nodes[0].score < 0.3
    ):

        def fallback_generator():

            fallback_message = (
                "I'm sorry, I don't have enough information "
                "in the company documents to answer that."
            )

            try:

                yield fallback_message

                trace.update(
                    output=fallback_message
                )

            finally:

                trace.end()
                langfuse.flush()

        return StreamingResponse(
            fallback_generator(),
            media_type="text/plain"
        )
        
    context_str = ""

    for node in reranked_nodes:

        context_str += (
            f"File Name: "
            f"{node.metadata.get('file_name')}\n"
        )

        context_str += (
            f"Content: "
            f"{node.text}\n\n"
        )

    prompt = f"""
        You are a helpful corporate assistant.

        Use ONLY the following context to answer the user's question.

        If the answer is not in the context, say:
        "I do not know."

        Do not use outside knowledge.

        At the end of your answer, cite the exact Source filenames
        you used.

        CHAT HISTORY:
        {history_str}

        CONTEXT:
        {context_str}

        QUESTION:
        {request.question}

        ANSWER:
        """

    def stream_generator():

        full_response = ""

        try:

            response = Settings.llm.stream_complete(prompt)

            for chunk in response:

                if chunk.delta:

                    full_response += chunk.delta

                    yield chunk.delta

            trace.update(
                output=full_response
            )

        except Exception as e:

            trace.update(
                output=f"Streaming error: {str(e)}"
            )

            raise

        finally:

            trace.end()
            langfuse.flush()
            
    return StreamingResponse(
        stream_generator(),
        media_type="text/plain"
    )

@app.get("/")
def api_status():

    return {
        "status": "API is running!"
    }