from llama_index.core import Settings
import qdrant_client
import time
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http import models as qmodels

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

reader = SimpleDirectoryReader("./data")
documents = reader.load_data()

for doc in documents:
    filename = doc.metadata.get("file_name", "").lower()
    
    if "handbook" in filename:
        doc.metadata["tenant_id"] = "company_A"
    elif "10-k" in filename or "report" in filename:
        doc.metadata["tenant_id"] = "company_B"
    else:
        doc.metadata["tenant_id"] = "unknown_document"
        
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
)

nodes = splitter.get_nodes_from_documents(documents)
print(f"Total nodes to index: {len(nodes)}")

client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333,
    grpc_port=6334,
    prefer_grpc=True,  
    timeout=600,        
)

collection_name = 'enterprise_docs'

if client.collection_exists(collection_name):
    print(f"Deleting broken collection '{collection_name}' (status was RED)...")
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=qmodels.VectorParams(
        size=768, 
        distance=qmodels.Distance.COSINE,
    ),
)

vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


BATCH_SIZE = 64
MAX_RETRIES = 3
 
index = VectorStoreIndex(nodes=[], storage_context=storage_context)
 
for i in range(0, len(nodes), BATCH_SIZE):
    batch = nodes[i : i + BATCH_SIZE]
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            index.insert_nodes(batch)
            print(f"Indexed {i + len(batch)}/{len(nodes)}")
            break
        except Exception as e:
            attempt += 1
            wait = 2 ** attempt
            print(f"Batch {i}-{i+len(batch)} failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(wait)

info = client.get_collection(collection_name)
print(f"Final status: {info.status}, points: {info.points_count}")