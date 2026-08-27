import os
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.schema import QueryBundle
import qdrant_client
from dotenv import load_dotenv

load_dotenv()


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

collection_name = "enterprise_docs"

vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
)

print("Existing documents loaded")

filters = MetadataFilters(
    filters=[MetadataFilter(key="tenant_id", value="company_A")]
)
query_engine = index.as_retriever(filters=filters, similarity_top_k=20)

cohere_api_key = os.environ["COHERE_API"]

cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=5)

question = QueryBundle(query_str="What are the pillars of the AWS framework?")

nodes = query_engine.retrieve(question)

print(nodes)

print("Reranking...")
reranked_nodes = cohere_rerank.postprocess_nodes(nodes, query_bundle=question)
print(len(reranked_nodes))

for i, node in enumerate(reranked_nodes, start=1):
    print("file_name:", node.metadata.get("file_name"))
    print("tenant_id:", node.metadata.get("tenant_id"))




