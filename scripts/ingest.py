import os
from llama_index.core import Settings
import qdrant_client
import time
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http import models as qmodels
from llama_index.readers.file import PyMuPDFReader
from dotenv import load_dotenv

load_dotenv()

Settings.embed_model = CohereEmbedding(
    cohere_api_key=os.environ["COHERE_API"], 
    model_name="embed-english-v3.0",
    input_type="search_document" 
)

parser = PyMuPDFReader()
file_extractor = {".pdf": parser}

documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

TENANT_MAPPING = {
    "wellarchitected-framework.pdf": "company_A",
    "the-economic-potential-of-generative-ai-the-next-productivity-frontier.pdf": "company_A",
    "nike-ar-2023.pdf": "company_B",
    "aapl-20230930.pdf": "company_B",
    "btpd_ee_handbook_2023.pdf": "company_A",
    "osage nation employee handbook (2023) - effective oct 16 2023_0.pdf": "company_A",
    "lakeside+employee+handbook++(oct+23).pdf": "company_A",
}

for doc in documents:
    filename = doc.metadata.get("file_name", "").strip().lower()
    
    if filename in TENANT_MAPPING:
        doc.metadata["tenant_id"] = TENANT_MAPPING[filename]
    else:
        doc.metadata["tenant_id"] = "unknown_document"
        
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
)

nodes = splitter.get_nodes_from_documents(documents)
print(f"Total nodes to index: {len(nodes)}")

client = qdrant_client.QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    port=6333,
    grpc_port=6334,
    prefer_grpc=True,
    timeout=600,
)

collection_name = 'enterprise_docs_v2'

if client.collection_exists(collection_name):
    print(f"Deleting broken collection '{collection_name}' (status was RED)...")
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=qmodels.VectorParams(
        size=1024, 
        distance=qmodels.Distance.COSINE,
    ),
)

print("Creating payload index for tenant_id...")
client.create_payload_index(
    collection_name=collection_name,
    field_name="tenant_id",
    field_schema=qmodels.PayloadSchemaType.KEYWORD,
)

vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


BATCH_SIZE = 16
MAX_RETRIES = 5
 
index = VectorStoreIndex(nodes=[], storage_context=storage_context)
 
for i in range(0, len(nodes), BATCH_SIZE):
    batch = nodes[i : i + BATCH_SIZE]
    attempt = 0
    
    while attempt < MAX_RETRIES:
        try:
            index.insert_nodes(batch)
            print(f"Successfully embedded and indexed {i + len(batch)}/{len(nodes)} chunks")
            # Sleep for a tiny bit between successful batches to pace ourselves
            time.sleep(2) 
            break
        except Exception as e:
            error_msg = str(e).lower()
            attempt += 1
            
            # If we hit the Cohere per-minute rate limit, wait a full 60 seconds!
            if "429" in error_msg or "too many requests" in error_msg or "rate limit" in error_msg:
                print(f"⚠️ Hit Cohere API rate limit! Sleeping for 60 seconds... (Attempt {attempt}/{MAX_RETRIES})")
                time.sleep(60)
            else:
                # For standard network glitches, just wait a few seconds
                wait = 2 ** attempt
                print(f"Batch {i}-{i+len(batch)} failed (attempt {attempt}/{MAX_RETRIES}): {e}")
                time.sleep(wait)
                
            if attempt == MAX_RETRIES:
                print("Max retries reached. Exiting script.")
                raise

info = client.get_collection(collection_name)
print(f"\nINGESTION COMPLETE! Final status: {info.status}, points: {info.points_count}")